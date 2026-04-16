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

# Importações do seu projeto
from src.dataset import build_memmap_and_stats, SegDatasetMemmap, build_gpu_augmenter
from src.utils import get_valid_files, compute_class_weights, evaluate_model
from src.model import AttentionResUNet
from src.metrics import *
from src.metrics import save_clean_plots  # Importando a função de plotagem

# Otimizações de performance para "torar" a GPU
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config):
    import time # Necessário para calcular o tempo da época
    
    # --- CARREGANDO CONFIGURAÇÕES ---
    paths = config['paths']
    ds_cfg = config['dataset']
    tr_cfg = config['training']

    os.makedirs(paths['checkpoint_path'], exist_ok=True)
    os.makedirs(paths['memmap_dir'], exist_ok=True)
    weights_path = os.path.join(paths['checkpoint_path'], 'class_weights.npy')

    # --- SETUP DE LOGGING ---
    log_file = os.path.join(paths['checkpoint_path'], f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.info('Iniciando treinamento...')

    # --- SPLIT E DATALOADERS ---
    all_files = get_valid_files(paths['images_dir'], paths['labels_dir'])
    train_files, val_files = train_test_split(all_files, test_size=ds_cfg['val_size'], random_state=42)
    
    img_size_tuple = tuple(ds_cfg['img_size'])
    
    train_img_mm, train_mask_mm, train_list, pmins, pmaxs, means, stds = build_memmap_and_stats(
        paths['images_dir'], paths['labels_dir'], paths['memmap_dir'], train_files,
        ds_cfg['num_bands'], img_size_tuple, split_name='train', compute_stats=True
    )
    val_img_mm, val_mask_mm, val_list, _, _, _, _ = build_memmap_and_stats(
        paths['images_dir'], paths['labels_dir'], paths['memmap_dir'], val_files, 
        ds_cfg['num_bands'], img_size_tuple, split_name='val', compute_stats=False
    )

    train_dataset = SegDatasetMemmap(train_img_mm, train_mask_mm, train_list, pmins, pmaxs, means, stds, ds_cfg['num_bands'], ds_cfg['crop_size'])
    val_dataset = SegDatasetMemmap(val_img_mm, val_mask_mm, val_list, pmins, pmaxs, means, stds, ds_cfg['num_bands'], ds_cfg['crop_size'])

    train_loader = DataLoader(train_dataset, batch_size=tr_cfg['batch_size'], shuffle=True, 
                              num_workers=ds_cfg['num_cores'], pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=tr_cfg['batch_size'], shuffle=False, 
                            num_workers=ds_cfg['num_cores'], pin_memory=True, persistent_workers=True)

    geometric, photometric = build_gpu_augmenter()
    
    # --- MODELO E OTIMIZADORES ---
    model = AttentionResUNet(ds_cfg['num_bands'], ds_cfg['num_classes']).to(device)
    model = model.to(memory_format=torch.channels_last)

    if os.path.exists(weights_path):
        class_weights = np.load(weights_path)
    else:
        class_weights = compute_class_weights(train_mask_mm, train_list, ds_cfg['crop_size'], ds_cfg['num_classes'])

    scaler = torch.amp.GradScaler('cuda')
    grad_accum = tr_cfg.get('grad_accum', 1)
    criterion = FocalDiceLoss(
        num_classes=ds_cfg['num_classes'],
        class_weights=class_weights,
        ignore_index=0,
    ).to(device)

    # Estatísticas de normalização na GPU (substituem o pré-processamento que era no __getitem__)
    device_pmins = train_dataset.pmins.to(device, non_blocking=True)
    device_pmaxs = train_dataset.pmaxs.to(device, non_blocking=True)
    device_means = train_dataset.means.to(device, non_blocking=True)
    device_stds  = train_dataset.stds.to(device, non_blocking=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=tr_cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr_cfg['epochs'])

    # --- HISTÓRICO ATUALIZADO (Adicionado lr, time e gpu_mem) ---
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_acc': [],
        'val_miou': [],
        'lr': [],        # Adicionado para a plotagem
        'time': [],      # Adicionado para a plotagem
        'gpu_mem': [],   # Adicionado para a plotagem
        'epoch': []
    }
    
    start_epoch = 0
    best_loss = float('inf')
    early_counter = 0

    # --- LOOP PRINCIPAL ---
    for epoch in range(start_epoch, tr_cfg['epochs']):
        epoch_start_time = time.time() # Marca o início da época
        
        # Reseta as estatísticas de memória da GPU no início da época
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        epoch_train_loss = 0.0
        optimizer.zero_grad()

        for i, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            # Normalização na GPU: clamp percentílico + z-score
            imgs = torch.clamp(imgs, device_pmins, device_pmaxs)
            imgs = (imgs - device_means) / (device_stds + 1e-6)

            with torch.no_grad():
                masks = masks.unsqueeze(1)
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

            epoch_train_loss += loss.item() * grad_accum

        # --- VALIDAÇÃO ---
        model.eval()
        epoch_val_loss, total_pacc, n = 0.0, 0.0, 0
        cm = np.zeros((ds_cfg['num_classes'], ds_cfg['num_classes']), dtype=np.int64)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                masks = masks.to(device, non_blocking=True)

                imgs = torch.clamp(imgs, device_pmins, device_pmaxs)
                imgs = (imgs - device_means) / (device_stds + 1e-6)

                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)

                preds = torch.argmax(outputs, dim=1)
                epoch_val_loss += loss.item()
                total_pacc += pixel_accuracy(preds, masks)
                n += 1

                pn, mn = preds.cpu().numpy().flatten(), masks.cpu().numpy().flatten()
                valid = (mn >= 0) & (mn < ds_cfg['num_classes'])
                cm += np.bincount(ds_cfg['num_classes'] * mn[valid] + pn[valid], 
                                  minlength=ds_cfg['num_classes']**2).reshape(ds_cfg['num_classes'], ds_cfg['num_classes'])

        # Cálculos de métricas da época
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / n
        avg_pacc = total_pacc / n
        
        # Cálculo de mIoU (Ignorando background na média final)
        ious = []
        for j in range(ds_cfg['num_classes']):
            intersection = cm[j, j]
            union = cm[j, :].sum() + cm[:, j].sum() - intersection
            ious.append(intersection / (union + 1e-6))
        avg_miou = np.nanmean(ious[1:]) 

        # Captura tempo, LR e memória GPU
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        gpu_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0

        # --- ATUALIZA HISTÓRICO COM TODAS AS CHAVES ---
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_pacc)
        history['val_miou'].append(avg_miou)
        history['lr'].append(current_lr)
        history['time'].append(epoch_duration)
        history['gpu_mem'].append(gpu_memory_gb)
        
        # Agora a função vai rodar sem KeyError processando todas as chaves exigidas
        save_clean_plots(history, "./metrics_unet/")

        logging.info(f"Época {epoch+1} Finalizada | ValLoss: {avg_val_loss:.4f} | mIoU: {avg_miou:.4f} | Tempo: {epoch_duration:.1f}s | GPU Mem: {gpu_memory_gb:.2f}GB")

        # Checkpoint e Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_counter = 0
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'best_loss': best_loss}, paths['best_model_path'])
        else:
            early_counter += 1
            if early_counter >= tr_cfg['patience']:
                logging.info("Early stopping acionado!")
                break
        
        scheduler.step()

    # --- AVALIAÇÃO FINAL ---
    evaluate_model(model, val_loader, ds_cfg['num_classes'], device, paths['csv_validation'], paths['checkpoint_path'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    train(config_dict)