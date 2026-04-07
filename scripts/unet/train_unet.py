import os
import argparse
import yaml
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import build_memmap_and_stats, SegDatasetMemmap, build_gpu_augmenter
from src.utils import get_valid_files, compute_class_weights, evaluate_model
from src.model import AttentionResUNet
from src.metrics import *

# Otimizações do PyTorch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

def train(config):
    # --- CARREGANDO CONFIGURAÇÕES ---
    paths = config['paths']
    ds_cfg = config['dataset']
    tr_cfg = config['training']

    # Variáveis de ambiente e caminhos
    os.makedirs(paths['checkpoint_path'], exist_ok=True)
    os.makedirs(paths['memmap_dir'], exist_ok=True)
    
    weights_path = os.path.join(paths['checkpoint_path'], 'class_weights.npy')

    # --- SETUP DE LOGGING ---
    log_file = os.path.join(paths['checkpoint_path'], f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.info('Iniciando script de treino via YAML.')

    # --- SPLIT DE DADOS ---
    all_files = get_valid_files(paths['images_dir'], paths['labels_dir'])
    train_files, val_files = train_test_split(all_files, test_size=ds_cfg['val_size'], random_state=42)
    logging.info(f"Total: {len(all_files)} | Treino: {len(train_files)} | Val: {len(val_files)}")

    # --- MEMMAP ---
    img_size_tuple = tuple(ds_cfg['img_size'])
    
    train_img_mm, train_mask_mm, train_list, pmins, pmaxs, means, stds = build_memmap_and_stats(
        paths['images_dir'], paths['labels_dir'], paths['memmap_dir'], train_files,
        ds_cfg['num_bands'], img_size_tuple, split_name='train', compute_stats=True
    )

    val_img_mm, val_mask_mm, val_list, _, _, _, _ = build_memmap_and_stats(
        paths['images_dir'], paths['labels_dir'], paths['memmap_dir'], val_files, 
        ds_cfg['num_bands'], img_size_tuple, split_name='val', compute_stats=False
    )

    # --- DATASETS E DATALOADERS ---
    train_dataset = SegDatasetMemmap(
        train_img_mm, train_mask_mm, train_list, pmins, pmaxs, means, stds, 
        ds_cfg['num_bands'], ds_cfg['crop_size']
    )
    val_dataset = SegDatasetMemmap(
        val_img_mm, val_mask_mm, val_list, pmins, pmaxs, means, stds, 
        ds_cfg['num_bands'], ds_cfg['crop_size']
    )

    train_loader = DataLoader(train_dataset, batch_size=tr_cfg['batch_size'], shuffle=True, 
                              num_workers=ds_cfg['num_cores'], pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=tr_cfg['batch_size'], shuffle=False, 
                            num_workers=ds_cfg['num_cores'], pin_memory=True, persistent_workers=True)

    geometric, photometric = build_gpu_augmenter()
    
    # --- MODELO ---
    model = AttentionResUNet(ds_cfg['num_bands'], ds_cfg['num_classes']).to(device)
    model = model.to(memory_format=torch.channels_last)

    # --- PESOS DE CLASSE ---
    if os.path.exists(weights_path):
        class_weights = np.load(weights_path)
    else:
        class_weights = compute_class_weights(mask_memmap_path=train_mask_mm, file_list=train_list, 
                                              crop_size=ds_cfg['crop_size'], num_classes=ds_cfg['num_classes'])
    class_weights[0] *= 0.1
    np.save(weights_path, class_weights)
    logging.info(f'Class weights: {class_weights}')

    scaler = torch.amp.GradScaler('cuda')
    early_counter = 0
    grad_accum = tr_cfg.get('grad_accum', 2) # Puxa do YAML, default 2

    criterion = FocalDiceLoss(num_classes=ds_cfg['num_classes'], class_weights=class_weights, gamma=1.0, dice_weight=1.0, focal_weight=1.0).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tr_cfg['learning_rate']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr_cfg['epochs'])

    start_epoch = 0
    best_loss = float('inf')

    # Retomar treino em caso de processo interrompido
    if os.path.exists(paths['best_model_path']):
        logging.info('Carregando checkpoint...')
        checkpoint = torch.load(paths['best_model_path'], map_location=device)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])

        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

        logging.info(f'Retomando da época {start_epoch} | best_loss={best_loss:.4f}')

    # --- LOOP DE TREINAMENTO ---
    for epoch in range(start_epoch, tr_cfg['epochs']):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Augmentação na GPU em batch
            with torch.no_grad():
                masks = masks.unsqueeze(1)  # (B,1,H,W)
                imgs, masks = geometric(imgs, masks)
                imgs = photometric(imgs)
                masks = masks.squeeze(1)

            with torch.amp.autocast(device_type='cuda'):
                loss = criterion(model(imgs), masks) / grad_accum

            scaler.scale(loss).backward()

            if (i+1) % grad_accum == 0 or (i+1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        total_pacc = 0.0
        n = 0
        cm = np.zeros((ds_cfg['num_classes'], ds_cfg['num_classes']), dtype=np.int64)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)

                preds = torch.argmax(outputs, dim=1)

                val_loss += loss.item()
                total_pacc += pixel_accuracy(preds, masks)
                n += 1

                pn = preds.cpu().numpy().flatten()
                mn = masks.cpu().numpy().flatten()

                valid = (mn >= 0) & (mn < ds_cfg['num_classes'])
                cm += np.bincount(ds_cfg['num_classes'] * mn[valid] + pn[valid], minlength=ds_cfg['num_classes']**2).reshape(ds_cfg['num_classes'], ds_cfg['num_classes'])

        val_loss /= n
        avg_pacc = total_pacc / n

        # Calcula mIoU
        ious = []
        for i in range(ds_cfg['num_classes']):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            denom = TP + FP + FN
            ious.append(TP / denom if denom > 0 else np.nan)

        avg_miou = np.nanmean(ious[1:])
        current_lr = scheduler.get_last_lr()[0]

        logging.info(f"Epoch {epoch+1:>4} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} "
                     f"| PixAcc: {avg_pacc:.4f} | mIoU: {avg_miou:.4f} | LR: {current_lr:.2e}")

        # Checkpoint & Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            early_counter = 0
            torch.save({
                'epoch': epoch, 
                'model_state': model.state_dict(), 
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss': best_loss
            }, paths['best_model_path'])
            logging.info(f" -> Melhor modelo salvo (val_loss={val_loss:.4f})")
        else:
            early_counter += 1
            if early_counter >= tr_cfg['patience']:
                logging.info(f"Early stopping acionado (sem melhoria por {tr_cfg['patience']} épocas).")
                break

    # --- AVALIAÇÃO FINAL ---
    logging.info("Iniciando avaliação final com o melhor modelo salvo.")
    checkpoint = torch.load(paths['best_model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    evaluate_model(model, val_loader, ds_cfg['num_classes'], device, paths['csv_validation'], paths['checkpoint_path'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    train(config_dict)