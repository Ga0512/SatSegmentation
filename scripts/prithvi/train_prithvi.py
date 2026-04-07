import os

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import kornia.augmentation as K

# Adicionando TorchMetrics para mIoU e Accuracy
from torchmetrics import JaccardIndex, Accuracy

from src.model import Prithvi11BandsModel
from src.dataset import PrithviDataset, build_memmap_and_stats_prithvi

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)
gdal.UseExceptions()

def train(config):
    # 1. CARREGANDO CONFIGURAÇÕES
    paths = config['paths']
    ds_cfg = config['dataset']
    tr_cfg = config['training']

    os.makedirs(paths['memmap_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Usando o dispositivo: {device}")

    # 2. PREPARAÇÃO DOS DADOS
    all_files = [f for f in os.listdir(paths['images_dir']) if f.lower().endswith('.tif')]
    img_p, mask_p, means, stds = build_memmap_and_stats_prithvi(
        paths['images_dir'], 
        paths['labels_dir'], 
        paths['memmap_dir'], 
        all_files, 
        ds_cfg['num_bands'], 
        tuple(ds_cfg['img_size']), 
        ds_cfg['crop_size']
    )
    
    total_samples = len(all_files) * (ds_cfg['img_size'][0] // ds_cfg['crop_size'])**2
    dataset = PrithviDataset(
        img_p, mask_p, total_samples, means, stds, 
        ds_cfg['num_bands'], ds_cfg['crop_size']
    )
    
    indices = np.arange(total_samples)
    train_idx, val_idx = train_test_split(indices, test_size=ds_cfg['val_size'], random_state=42)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=tr_cfg['batch_size'], shuffle=True, num_workers=ds_cfg['num_cores'])
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=tr_cfg['batch_size'], num_workers=ds_cfg['num_cores'])

    # 3. INICIALIZAÇÃO DO MODELO
    model = Prithvi11BandsModel(ds_cfg['num_classes'], ds_cfg['num_bands']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=tr_cfg['learning_rate'], weight_decay=tr_cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Inicializando Métricas
    metric_miou = JaccardIndex(task="multiclass", num_classes=ds_cfg['num_classes'], ignore_index=0).to(device)
    metric_acc = Accuracy(task="multiclass", num_classes=ds_cfg['num_classes'], ignore_index=0).to(device)

    aug = K.AugmentationSequential(K.RandomHorizontalFlip(), K.RandomVerticalFlip(), data_keys=['input', 'mask']).to(device)

    best_loss = float('inf')
    epochs_no_improve = 0

    # 4. LOOP DE TREINAMENTO
    for epoch in range(tr_cfg['epochs']):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Época {epoch+1}/{tr_cfg['epochs']}"):
            x, y = x.to(device), y.to(device)
            x, y = aug(x, y)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y.squeeze().long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validação
        model.eval()
        val_loss = 0
        metric_miou.reset()
        metric_acc.reset()
        
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device).long()
                v_logits = model(vx)
                val_loss += criterion(v_logits, vy).item()
                
                metric_miou.update(v_logits, vy)
                metric_acc.update(v_logits, vy)
        
        final_miou = metric_miou.compute()
        final_acc = metric_acc.compute()
        avg_v_loss = val_loss / len(val_loader)
        
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {avg_v_loss:.4f} | mIoU: {final_miou:.4f} | Pixel Acc: {final_acc:.4f}")

        # Checkpoint & Early Stopping baseado em Val Loss
        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), paths['save_model_path'])
            print(">>> Modelo salvo (Melhor Val Loss)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= tr_cfg['patience']:
                print(f"Interrompendo: Sem melhoria por {tr_cfg['patience']} épocas.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    train(config_dict)