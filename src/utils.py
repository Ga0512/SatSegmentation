import os
import torch
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_valid_files(images_dir, labels_dir, mask_suffix='_mask'):
    label_map = {os.path.splitext(f)[0]: f for f in os.listdir(labels_dir) if f.lower().endswith('.tif')}
    valid_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.tif') and os.path.splitext(f)[0] + mask_suffix in label_map]

    return valid_files


def compute_patch_class_distribution(mask_memmap_path, total_samples, indices, crop_size, num_classes):
    """
    Para cada patch em `indices`, retorna a fração de pixels de cada classe
    (após filtrar NoData). Shape: [len(indices), num_classes], float32 em [0,1].

    Usado para oversampling proporcional à QUANTIDADE de classe rara no patch
    (e não só à presença binária).
    """
    masks = np.memmap(mask_memmap_path, dtype='int16', mode='r',
                      shape=(total_samples, crop_size, crop_size))
    distribution = np.zeros((len(indices), num_classes), dtype=np.float32)
    for i, idx in enumerate(tqdm(indices, desc='Distribuição de classes por patch')):
        flat = masks[idx].reshape(-1).astype(np.int64)
        valid = flat[(flat >= 0) & (flat < num_classes)]
        if valid.size > 0:
            counts = np.bincount(valid, minlength=num_classes)
            distribution[i] = counts / valid.size
    return distribution


def compute_class_weights(mask_memmap_path, file_list, crop_size, num_classes):
    N = len(file_list)
    masks = np.memmap(mask_memmap_path, dtype='int16', mode='r', shape=(N, crop_size, crop_size))

    class_counts = np.zeros(num_classes, dtype=np.int64)
    batch = 256

    for i in tqdm(range(0, N, batch), desc='Contando pixels'):
        flat = masks[i:i+batch].reshape(-1).astype(np.int64)
        # Filtra NoData (255) e valores fora de [0, num_classes); class 0 é background válido.
        valid = flat[(flat >= 0) & (flat < num_classes)]
        class_counts += np.bincount(valid, minlength=num_classes)

    total = max(class_counts.sum(), 1)
    freq = class_counts / total
    weights = np.log(1.0 / (freq + 1e-8))
    weights = np.clip(weights, 0.2, 10.0)

    # Normaliza para média 1.0 entre todas as classes válidas (incl. background).
    weights = weights / max(weights.mean(), 1e-6)

    # Diagnóstico: distribuição de pixels e pesos finais por classe
    logging.info('Distribuição de pixels por classe:')
    for c in range(num_classes):
        pct = 100.0 * freq[c]
        logging.info(f'  classe {c}: {class_counts[c]:>12,} pixels ({pct:6.3f}%)  weight={weights[c]:.3f}')

    return weights.astype(np.float32)


def evaluate_model(model, dataloader, num_classes, device, csv_path, log_dir):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Acumula a confusion matrix
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast(device_type='cuda'):
                preds = torch.argmax(model(imgs), dim=1)

            pn = preds.cpu().numpy().flatten()
            mn = masks.cpu().numpy().flatten()
            valid = (mn >= 0) & (mn < num_classes)
            cm += np.bincount(num_classes * mn[valid] + pn[valid], minlength=num_classes**2).reshape(num_classes, num_classes)

    # ----- Métricas globais -----
    total = cm.sum()
    pix_acc = np.trace(cm) / total if total > 0 else 0.0

    # IoU por classe
    ious = []
    for i in range(num_classes):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        denom = TP + FP + FN
        ious.append(TP/denom if denom > 0 else np.nan)
    miou = np.nanmean(ious)

    # Kappa de Cohen
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / total if total > 0 else np.zeros_like(cm)
    kappa = (np.trace(cm) - np.trace(expected)) / (total - np.trace(expected)) if total - np.trace(expected) != 0 else 0

    # Dice por classe
    dice = []
    for i in range(num_classes):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        denom = 2*TP + FP + FN
        dice.append(2*TP/denom if denom > 0 else np.nan)

    # Weighted IoU
    w = cm.sum(axis=1) / total
    wiou = np.nansum(np.array(ious) * w)

    # Métricas de precisão/recall/F1
    metrics = {}
    for i in range(num_classes):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        precision = TP/(TP+FP) if TP+FP > 0 else 0
        recall    = TP/(TP+FN) if TP+FN > 0 else 0
        f1        = 2*precision*recall/(precision+recall) if precision+recall > 0 else 0
        metrics[i] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': TP+FN}

    # Salva CSV
    df = pd.DataFrame(metrics).T
    df['Pixel Accuracy'] = pix_acc
    df['Mean IoU']       = miou
    df['Kappa']          = kappa
    df['Dice']           = dice
    df['Weighted IoU']   = wiou
    df.to_csv(csv_path)

    # Matriz de confusão
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(log_dir,'confusion_matrix.png'))
    plt.close()

    logging.info(f'Pixel Acc: {pix_acc:.4f} | mIoU: {miou:.4f} | Kappa: {kappa:.4f}')
    return pix_acc, miou, kappa, dice, wiou

