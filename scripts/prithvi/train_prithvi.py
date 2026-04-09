import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import kornia.augmentation as K

from torchmetrics import JaccardIndex, Accuracy

from src.model import Prithvi11BandsModel
from src.dataset import PrithviDataset, build_memmap_and_stats_prithvi
from src.metrics import save_clean_plots

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)
gdal.UseExceptions()

def train(config):
    # 1. CARREGANDO CONFIGURAÇÕES
    paths = config['paths']
    ds_cfg = config['dataset']
    tr_cfg = config['training']

    os.makedirs(paths['memmap_dir'], exist_ok=True)
    
    # --- PASTA DE MÉTRICAS ---
    metrics_dir = os.path.join(os.path.dirname(paths.get('save_model_path', './')), 'metrics_prithvi')
    os.makedirs(metrics_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler('cuda')

    logging.info(f"Usando o dispositivo: {device}")

    # 2. PREPARAÇÃO DOS DADOS
    all_files = [f for f in os.listdir(paths['images_dir']) if f.lower().endswith('.tif')]
    img_p, mask_p, means, stds = build_memmap_and_stats_prithvi(
        paths['images_dir'], paths['labels_dir'], paths['memmap_dir'], 
        all_files, ds_cfg['num_bands'], tuple(ds_cfg['img_size']), ds_cfg['crop_size']
    )
    
    total_samples = len(all_files) * (ds_cfg['img_size'][0] // ds_cfg['crop_size'])**2
    dataset = PrithviDataset(
        img_p, mask_p, total_samples, means, stds, 
        ds_cfg['num_bands'], ds_cfg['crop_size']
    )
    
    indices = np.arange(total_samples)
    train_idx, val_idx = train_test_split(indices, test_size=ds_cfg['val_size'], random_state=42)
    
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=tr_cfg['batch_size'], 
        shuffle=True, num_workers=ds_cfg['num_cores'],
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=tr_cfg['batch_size'], num_workers=ds_cfg['num_cores'])

    # 3. INICIALIZAÇÃO DO MODELO E MÉTRICAS
    model = Prithvi11BandsModel(ds_cfg['num_classes'], ds_cfg['num_bands']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=tr_cfg['learning_rate'], weight_decay=tr_cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    metric_miou = JaccardIndex(task="multiclass", num_classes=ds_cfg['num_classes'], ignore_index=0).to(device)
    metric_acc = Accuracy(task="multiclass", num_classes=ds_cfg['num_classes'], ignore_index=0).to(device)

    aug = K.AugmentationSequential(K.RandomHorizontalFlip(), K.RandomVerticalFlip(), data_keys=['input', 'mask']).to(device)

    best_loss = float('inf')
    epochs_no_improve = 0
    
    # Dicionário para armazenar o histórico
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_acc': [], 'lr': [], 'time': [], 'gpu_mem': []}

    try:
        for epoch in range(tr_cfg['epochs']):
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats() # Reseta o contador de VRAM
            
            model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Época {epoch+1}")
            
            for x, y in pbar:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                x, y = aug(x, y)
                
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, y.squeeze().long())
                
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Calcula loss média de treino
            avg_t_loss = train_loss / len(train_loader)

            # --- VALIDAÇÃO ---
            model.eval()
            val_loss = 0
            metric_miou.reset()
            metric_acc.reset()
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                for vx, vy in val_loader:
                    vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True).long()
                    
                    v_logits = model(vx)
                    val_loss += criterion(v_logits, vy).item()
                    
                    metric_miou.update(v_logits, vy)
                    metric_acc.update(v_logits, vy)
            
            final_miou = metric_miou.compute().item()
            final_acc = metric_acc.compute().item()
            avg_v_loss = val_loss / len(val_loader)
            
            # --- COLETA DE MÉTRICAS GERAIS ---
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            # Pega o pico de memória alocada na GPU em Gigabytes (GB)
            gpu_mem = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0
            
            # Salvando no dicionário
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_t_loss)
            history['val_loss'].append(avg_v_loss)
            history['val_miou'].append(final_miou)
            history['val_acc'].append(final_acc)
            history['lr'].append(current_lr)
            history['time'].append(epoch_time)
            history['gpu_mem'].append(gpu_mem)

            # Checkpoint inteligente
            if avg_v_loss < best_loss:
                best_loss = avg_v_loss
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, paths['save_model_path'])
                print(f"[*] Melhoria detectada. Modelo salvo. mIoU: {final_miou:.4f} PACC: {final_acc:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= tr_cfg['patience']:
                    print(f"[!] Early stopping at epoch {epoch+1}")
                    break
    
    finally:
        # Garante que os gráficos serão gerados mesmo se o treino for interrompido manualmente (Ctrl+C)
        logging.info("Salvando gráficos de métricas...")
        save_clean_plots(history, metrics_dir)
        logging.info(f"Gráficos salvos com sucesso na pasta: {metrics_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    train(config_dict)