import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def save_clean_plots(history, metrics_dir):
    """Gera e salva gráficos limpos para cada métrica monitorada."""
    # Usando um estilo limpo nativo do matplotlib
    plt.style.use('seaborn-v0_8-whitegrid')
    
    epochs = history['epoch']
    
    plots_config = [
        ('loss.png', 'Evolução da Loss', 'Loss', [
            (history['train_loss'], 'Train Loss', '#1f77b4'), 
            (history['val_loss'], 'Val Loss', '#ff7f0e')
        ]),
        ('accuracy_miou.png', 'Métricas de Validação (Acc & mIoU)', 'Score', [
            (history['val_acc'], 'Acurácia', '#2ca02c'), 
            (history['val_miou'], 'mIoU', '#d62728')
        ]),
        ('learning_rate.png', 'Learning Rate por Época', 'LR', [
            (history['lr'], 'Learning Rate', '#9467bd')
        ]),
        ('tempo_epoca.png', 'Tempo de Execução por Época', 'Segundos', [
            (history['time'], 'Tempo (s)', '#8c564b')
        ]),
        ('uso_gpu.png', 'Pico de Uso de Memória da GPU', 'Gigabytes (GB)', [
            (history['gpu_mem'], 'Memória GPU (GB)', '#e377c2')
        ])
    ]

    for filename, title, ylabel, lines in plots_config:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for data, label, color in lines:
            ax.plot(epochs, data, label=label, color=color, linewidth=2, marker='o', markersize=4)
            
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel('Épocas', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Ajustes para deixar o gráfico mais "limpo" (sem poluição)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Força o eixo X a mostrar apenas números inteiros (épocas)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        ax.legend(frameon=True, fancybox=True, shadow=False)
        plt.tight_layout()
        
        save_path = os.path.join(metrics_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        pt        = torch.exp(log_probs).gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt    = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss      = -(1 - pt) ** self.gamma * log_pt
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, gamma=1.0, dice_weight=1.0, focal_weight=1.0, eps=1e-6):
        """
        Loss combinada: Focal + Dice para segmentação multi-classe

        Args:
            num_classes: número de classes (int)
            class_weights: array numpy ou tensor com pesos por classe
            gamma: parâmetro gamma da focal loss
            dice_weight: peso da Dice Loss
            focal_weight: peso da Focal Loss
            eps: pequena constante para estabilidade numérica
        """
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.eps = eps

        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = torch.ones(num_classes, dtype=torch.float32)

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor [B, C, H, W], saída bruta do modelo
            targets: Tensor [B, H, W], índices das classes
        """
        B, C, H, W = logits.shape
        device = logits.device

        # ==== FOCAL LOSS ====
        log_probs = F.log_softmax(logits, dim=1)                  # [B,C,H,W]
        probs     = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets, num_classes=C).permute(0,3,1,2).float()  # [B,C,H,W]

        pt = (probs * targets_onehot).sum(1)                     # [B,H,W]
        log_pt = (log_probs * targets_onehot).sum(1)

        alpha = self.class_weights.to(device)
        alpha_factor = (targets_onehot * alpha.view(1,C,1,1)).sum(1)   # [B,H,W]

        focal_loss = - alpha_factor * ((1 - pt) ** self.gamma) * log_pt
        focal_loss = focal_loss.mean()

        # ==== DICE LOSS ====
        probs_flat = probs.reshape(B, C, -1)                     # [B,C,H*W]
        targets_flat = targets_onehot.reshape(B, C, -1)          # [B,C,H*W]

        intersection = (probs_flat * targets_flat).sum(-1)
        union = probs_flat.sum(-1) + targets_flat.sum(-1)
        dice_loss = 1 - (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = dice_loss.mean()

        # ==== COMBINAÇÃO ====
        loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return loss

def pixel_accuracy(preds, targets):
    return (preds == targets).sum().item() / targets.numel()

def batch_miou(preds, targets, num_classes):
    preds, targets = preds.view(-1), targets.view(-1)
    ious = []
    for c in range(num_classes):
        TP = ((preds==c) & (targets==c)).sum().item()
        FP = ((preds==c) & (targets!=c)).sum().item()
        FN = ((preds!=c) & (targets==c)).sum().item()
        d  = TP + FP + FN
        ious.append(TP/d if d > 0 else np.nan)
    return np.nanmean(ious)

def classification_metrics_from_confmatrix(cm):
    metrics = {}
    for i in range(cm.shape[0]):
        TP = cm[i,i]; FP = cm[:,i].sum()-TP; FN = cm[i,:].sum()-TP
        p  = TP/(TP+FP) if TP+FP>0 else 0
        r  = TP/(TP+FN) if TP+FN>0 else 0
        f1 = 2*p*r/(p+r) if p+r>0 else 0
        metrics[i] = {'precision':p,'recall':r,'f1-score':f1,'support':TP+FN}
    return metrics

def dice_per_class(cm):
    scores = []
    for i in range(cm.shape[0]):
        TP=cm[i,i]; FP=cm[:,i].sum()-TP; FN=cm[i,:].sum()-TP
        d=2*TP+FP+FN
        scores.append(2*TP/d if d>0 else np.nan)
    return scores

def weighted_iou(cm):
    total  = cm.sum()
    w      = cm.sum(axis=1) / total
    ious   = []
    for i in range(cm.shape[0]):
        TP=cm[i,i]; FP=cm[:,i].sum()-TP; FN=cm[i,:].sum()-TP
        d=TP+FP+FN
        ious.append(TP/d if d>0 else np.nan)
    return np.nansum(np.array(ious)*w)
