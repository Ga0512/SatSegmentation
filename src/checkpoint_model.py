import torch
import logging

def load_checkpoint(path, model, optimizer, scheduler, device):
    """
    Carrega um checkpoint completo e retorna o estado de treino restaurado.

    Suporta dois formatos:
      - Checkpoint completo (salvo por este script): contém epoch, optimizer,
        scheduler, history, best_loss, epochs_no_improve.
      - Checkpoint de pesos apenas (ex: pesos pré-treinados externos): contém
        somente model_state_dict. O treino recomeça do epoch 0 com estado limpo.

    Retorna:
        start_epoch (int): próximo epoch a executar.
        best_loss (float): melhor val_loss registrado até agora.
        epochs_no_improve (int): contador de early stopping.
        history (dict): histórico de métricas para plots contínuos.
    """
    logging.info(f"Carregando checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    # --- Pesos do modelo (obrigatório em qualquer formato) ---
    model.load_state_dict(ckpt['model_state_dict'])

    is_full_checkpoint = 'optimizer_state_dict' in ckpt

    if is_full_checkpoint:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        start_epoch      = ckpt.get('epoch', 0) + 1
        best_loss        = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        history          = ckpt.get('history', _empty_history())

        logging.info(
            f"Retomando do epoch {start_epoch} | "
            f"best_val_loss={best_loss:.4f} | "
            f"epochs_sem_melhora={epochs_no_improve}"
        )
    else:
        # Apenas pesos pré-treinados — treino começa do zero
        start_epoch       = 0
        best_loss         = float('inf')
        epochs_no_improve = 0
        history           = _empty_history()
        logging.info("Pesos pré-treinados carregados. Treino iniciará do epoch 0.")

    return start_epoch, best_loss, epochs_no_improve, history


def _empty_history():
    return {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'val_miou': [], 'val_acc': [], 'lr': [], 'time': [], 'gpu_mem': []
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_loss, epochs_no_improve, history):
    """Salva checkpoint completo — suficiente para retomar o treino exatamente."""
    torch.save({
        'epoch':               epoch,
        'model_state_dict':    model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss':           best_loss,
        'epochs_no_improve':   epochs_no_improve,
        'history':             history,
    }, path)

