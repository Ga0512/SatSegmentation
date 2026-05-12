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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import kornia.augmentation as K
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchmetrics import JaccardIndex, Accuracy

from src.model import Prithvi11BandsModel
from src.dataset import PrithviDataset, build_memmap_and_stats_prithvi, build_gpu_augmenter
from src.metrics import save_clean_plots, FocalDiceLoss
from src.checkpoint_model import load_checkpoint, _empty_history, save_checkpoint
from src.utils import compute_class_weights, compute_patch_class_distribution

logging.basicConfig(level=logging.INFO)


def train(config, resume_path=None):
    # ── 1. CONFIGURAÇÕES ──────────────────────────────────────────────────────
    paths  = config['paths']
    ds_cfg = config['dataset']
    tr_cfg = config['training']

    os.makedirs(paths['memmap_dir'], exist_ok=True)

    metrics_dir = os.path.join(
        os.path.dirname(paths.get('save_model_path', './')), 'metrics_prithvi'
    )
    os.makedirs(metrics_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler('cuda')

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    logging.info(f"Dispositivo: {device}")

    # ── 2. DADOS ──────────────────────────────────────────────────────────────
    all_files = [f for f in os.listdir(paths['images_dir']) if f.lower().endswith('.tif')]
    img_p, mask_p, means, stds = build_memmap_and_stats_prithvi(
        paths['images_dir'], paths['labels_dir'], paths['memmap_dir'],
        all_files, ds_cfg['num_bands'], tuple(ds_cfg['img_size']), ds_cfg['crop_size'],
        num_classes=ds_cfg['num_classes'],
    )

    total_samples = len(all_files) * (ds_cfg['img_size'][0] // ds_cfg['crop_size']) ** 2
    dataset = PrithviDataset(
        img_p, mask_p, total_samples, means, stds,
        ds_cfg['num_bands'], ds_cfg['crop_size']
    )

    indices = np.arange(total_samples)
    train_idx, val_idx = train_test_split(indices, test_size=ds_cfg['val_size'], random_state=42)

    # Pesos por classe (precisa vir antes do sampler)
    class_weights = compute_class_weights(
        mask_p, list(range(total_samples)),
        ds_cfg['crop_size'], ds_cfg['num_classes']
    )

    # Oversampling pela RARIDADE EM PATCHES da classe (não pela fração de pixels).
    # Para cada patch i: peso = max_c (boost[c] se classe c presente em i)
    # onde boost[c] = total_patches / patches_que_contêm_c
    # Patches com classes raras "puxam" o sampling; o background não dá boost.
    use_oversampling = tr_cfg.get('oversample_rare_classes', True)
    if use_oversampling:
        train_dist = compute_patch_class_distribution(
            mask_p, total_samples, train_idx.tolist(),
            ds_cfg['crop_size'], ds_cfg['num_classes']
        )
        presence = (train_dist > 0).astype(np.float32)
        patches_with_class = presence.sum(0)
        logging.info(f"Patches contendo cada classe (treino): "
                     f"{ {i: int(patches_with_class[i]) for i in range(ds_cfg['num_classes'])} }")

        N_train = len(train_idx)
        boost = N_train / np.maximum(patches_with_class, 1.0)   # [C]
        boost[0] = 1.0  # background nunca puxa o sampler

        logging.info(f"Boost de raridade por classe (p/ sampler): "
                     f"{ {i: float(f'{boost[i]:.2f}') for i in range(ds_cfg['num_classes'])} }")

        sampler_weights = (presence * boost[None, :]).max(axis=1)
        sampler_weights = np.clip(sampler_weights, 1e-3, None)

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sampler_weights).double(),
            num_samples=N_train,
            replacement=True,
        )
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=tr_cfg['batch_size'],
            sampler=sampler, num_workers=ds_cfg['num_cores'],
            pin_memory=True, persistent_workers=True, prefetch_factor=ds_cfg['num_cores'], drop_last=True
        )
    else:
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=tr_cfg['batch_size'],
            shuffle=True, num_workers=ds_cfg['num_cores'],
            pin_memory=True, persistent_workers=True, prefetch_factor=ds_cfg['num_cores'], drop_last=True
        )

    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=tr_cfg['batch_size'],
        num_workers=ds_cfg['num_cores'],
        pin_memory=True, persistent_workers=True, prefetch_factor=ds_cfg['num_cores']
    )

    # ── 3. MODELO, OTIMIZADOR, SCHEDULER ─────────────────────────────────────
    model = Prithvi11BandsModel(
        ds_cfg['num_classes'],
        ds_cfg['num_bands'],
        model_size=config.get('model_size', 'tiny'),
        model_bands=ds_cfg.get('model_bands', None),
    ).to(device)
    # LR diferenciado: backbone pretreinado (ViT blocks) usa LR baixo; patch_embed novo,
    # decoder e head (frescos) usam LR maior. Padrão em fine-tuning de ViT.
    backbone_params, decoder_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'encoder' in n and 'patch_embed' not in n:
            backbone_params.append(p)
        else:
            decoder_params.append(p)

    backbone_lr = tr_cfg['learning_rate']
    decoder_lr  = tr_cfg.get('decoder_lr', backbone_lr * 10)
    logging.info(f"Param groups: backbone={len(backbone_params)} (lr={backbone_lr:.1e}) | "
                 f"decoder/head={len(decoder_params)} (lr={decoder_lr:.1e})")

    optimizer = optim.AdamW(
        [{'params': backbone_params, 'lr': backbone_lr},
         {'params': decoder_params,  'lr': decoder_lr}],
        weight_decay=tr_cfg['weight_decay'],
    )
    
    # Defina quantas épocas de warmup você quer (geralmente entre 5% a 10% do total de épocas)
    warmup_epochs = 5 
    
    # 1. Scheduler de Warmup: Começa com 1% do LR do optimizer e sobe até 100% linearmente
    scheduler_warmup = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )

    # 2. Scheduler Principal (Cosseno): Age no restante das épocas
    # Note que o T_max agora é o total de épocas MENOS as épocas de warmup
    # Adicionei um eta_min para o LR não chegar a zero absoluto no final
    scheduler_cosine = CosineAnnealingLR(
        optimizer, 
        T_max=tr_cfg['epochs'] - warmup_epochs, 
        eta_min=1e-6 
    )

    # 3. Junta tudo em sequência
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_epochs] # O momento exato (época) onde ocorre a troca
    )

    criterion = FocalDiceLoss(
        num_classes=ds_cfg['num_classes'],
        class_weights=class_weights,
        gamma=tr_cfg.get('focal_gamma', 2.0),
        focal_weight=tr_cfg.get('focal_weight', 1.0),
        dice_weight=tr_cfg.get('dice_weight', 1.0),
        ignore_index=255,
    ).to(device)

    metric_miou = JaccardIndex(
        task="multiclass", num_classes=ds_cfg['num_classes'], ignore_index=255
    ).to(device)
    # mIoU por classe — diagnóstico pra detectar colapso na classe majoritária
    metric_miou_pc = JaccardIndex(
        task="multiclass", num_classes=ds_cfg['num_classes'], ignore_index=255,
        average=None,
    ).to(device)
    metric_acc = Accuracy(
        task="multiclass", num_classes=ds_cfg['num_classes'], ignore_index=255
    ).to(device)

    geometric, photometric = build_gpu_augmenter()

    # ── 4. RESUME ─────────────────────────────────────────────────────────────
    # Resolve o caminho: --resume pode ser o path explícito ou auto (usa save_model_path)
    if resume_path is True or resume_path == 'auto':
        resume_path = paths['save_model_path']

    if resume_path and os.path.isfile(resume_path):
        start_epoch, best_loss, epochs_no_improve, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
    else:
        if resume_path:
            logging.warning(f"Checkpoint não encontrado: {resume_path}. Iniciando do zero.")
        start_epoch       = 0
        best_loss         = float('inf')
        epochs_no_improve = 0
        history           = _empty_history()

    # ── 5. LOOP DE TREINO ─────────────────────────────────────────────────────
    device_means = dataset.means.to(device, non_blocking=True)
    device_stds  = dataset.stds.to(device, non_blocking=True)

    try:
        for epoch in range(start_epoch, tr_cfg['epochs']):
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            model.train()
            train_loss = 0

            pbar = tqdm(train_loader, desc=f"Época {epoch + 1}/{tr_cfg['epochs']}")

            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                x = (x - device_means) / (device_stds + 1e-6)

                with torch.no_grad():
                    y = y.unsqueeze(1).float()
                    x, y = geometric(x, y)
                    x = photometric(x)
                    y = y.squeeze(1).long()

                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss   = criterion(logits, y.squeeze().long())

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_t_loss = train_loss / len(train_loader)

            # ── Validação ───────────────────────────────────────────────────
            model.eval()
            val_loss = 0
            metric_miou.reset()
            metric_miou_pc.reset()
            metric_acc.reset()

            with torch.no_grad(), torch.amp.autocast('cuda'):
                for vx, vy in val_loader:
                    vx = vx.to(device, non_blocking=True)
                    vy = vy.to(device, non_blocking=True).long()
                    vx = (vx - device_means) / (device_stds + 1e-6)

                    v_logits  = model(vx)

                    # Remove a dimensão extra de canal da máscara
                    vy_squeezed = vy.squeeze(1).long() if vy.dim() == 4 else vy.long()

                    val_loss += criterion(v_logits, vy_squeezed).item()
                    metric_miou.update(v_logits, vy_squeezed)
                    metric_miou_pc.update(v_logits, vy_squeezed)
                    metric_acc.update(v_logits, vy_squeezed)

            final_miou = metric_miou.compute().item()
            final_miou_pc = metric_miou_pc.compute().detach().cpu().numpy()
            final_acc  = metric_acc.compute().item()
            avg_v_loss = val_loss / len(val_loader)

            pc_str = " ".join(f"c{i}={iou:.3f}" for i, iou in enumerate(final_miou_pc))
            logging.info(f"[Epoch {epoch+1}] mIoU per class: {pc_str}")

            scheduler.step()

            epoch_time  = time.time() - start_time
            current_lr  = scheduler.get_last_lr()[0]
            gpu_mem     = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if torch.cuda.is_available() else 0.0
            )

            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_t_loss)
            history['val_loss'].append(avg_v_loss)
            history['val_miou'].append(final_miou)
            history['val_acc'].append(final_acc)
            history['lr'].append(current_lr)
            history['time'].append(epoch_time)
            history['gpu_mem'].append(gpu_mem)

            logging.info(
                f"[Epoch {epoch+1}] train_loss={avg_t_loss:.4f} "
                f"val_loss={avg_v_loss:.4f} mIoU={final_miou:.4f} "
                f"acc={final_acc:.4f} lr={current_lr:.2e} "
                f"t={epoch_time:.1f}s mem={gpu_mem:.2f}GB"
            )

            # ── Checkpoint ──────────────────────────────────────────────────
            if avg_v_loss < best_loss:
                best_loss         = avg_v_loss
                epochs_no_improve = 0
                save_checkpoint(
                    paths['save_model_path'],
                    model, optimizer, scheduler,
                    epoch, best_loss, epochs_no_improve, history
                )
                print(
                    f"[*] Melhoria detectada → checkpoint salvo. "
                    f"mIoU={final_miou:.4f}  PACC={final_acc:.4f}"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= tr_cfg['patience']:
                    print(f"[!] Early stopping no epoch {epoch + 1}")
                    break

    finally:
        logging.info("Salvando gráficos de métricas...")
        save_clean_plots(history, metrics_dir)
        logging.info(f"Gráficos salvos em: {metrics_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento Prithvi")
    parser.add_argument("--config", type=str, required=True,
                        help="Caminho para o config.yaml")
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        metavar="CHECKPOINT_PATH",
        help=(
            "Retoma ou usa como pré-treino a partir de um checkpoint.\n"
            "  --resume          → usa paths.save_model_path do config\n"
            "  --resume path/ck  → usa o caminho explícito\n"
            "  (omitido)         → começa do zero"
        )
    )
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    train(config_dict, resume_path=args.resume)