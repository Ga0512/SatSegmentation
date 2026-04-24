"""
Benchmark completo: treino + inferência dos 4 modelos.

  python -m scripts.benchmark --all          # treina tudo + compara
  python -m scripts.benchmark                # compara checkpoints já existentes
  python -m scripts.benchmark --all --batch-size 2

Variantes:
  Prithvi Tiny  |  Prithvi 100M  |  UNet ResNet34  |  UNet ResNet50

Resultados salvos em ./benchmark/<variante>/
"""

import argparse
import copy
import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np

BENCH_DIR = './benchmark'

VARIANTS = [
    ('prithvi', 'tiny',     'Prithvi-Tiny'),
    ('prithvi', '100m',     'Prithvi-100M'),
    ('unet',    'resnet34', 'UNet-R34'),
    ('unet',    'resnet50', 'UNet-R50'),
]


# ── config helpers ────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_variant_config(kind, model_size, base_cfg):
    """Clona o config base e aponta paths para o diretório de benchmark."""
    cfg  = copy.deepcopy(base_cfg)
    cfg['model_size'] = model_size
    vdir = os.path.join(BENCH_DIR, f'{kind}_{model_size}')
    os.makedirs(vdir, exist_ok=True)

    if kind == 'prithvi':
        cfg['paths']['save_model_path'] = os.path.join(vdir, 'checkpoint.pth')
        # metrics_dir é derivado de save_model_path em train_prithvi
    else:
        cfg['paths']['checkpoint_path'] = vdir + os.sep
        cfg['paths']['best_model_path'] = os.path.join(vdir, 'checkpoint.pth')
        cfg['paths']['csv_validation']  = os.path.join(vdir, 'validation.csv')

    return cfg, vdir


def ckpt_path(cfg, kind):
    p = cfg.get('paths', {})
    return p.get('save_model_path') if kind == 'prithvi' else p.get('best_model_path')


# ── treino ────────────────────────────────────────────────────────────────────

def run_training(kind, cfg):
    if kind == 'prithvi':
        from scripts.prithvi.train_prithvi import train
        train(cfg)
    else:
        from scripts.unet.train_unet import train
        train(cfg)


# ── métricas dos checkpoints ──────────────────────────────────────────────────

def load_history(path):
    if not path or not os.path.isfile(path):
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck.get('history', None)


def extract_metrics(history):
    if not history or not history.get('val_miou'):
        return {}

    miou_arr = np.array(history['val_miou'])
    loss_arr = np.array(history['val_loss'])
    im = int(np.argmax(miou_arr))
    il = int(np.argmin(loss_arr))

    ep = history.get('epoch', list(range(1, len(miou_arr) + 1)))

    return {
        'best_miou':        miou_arr[im],
        'best_miou_epoch':  ep[im],
        'best_acc':         np.array(history['val_acc'])[im] if history.get('val_acc') else None,
        'best_val_loss':    loss_arr[il],
        'best_loss_epoch':  ep[il],
        'final_miou':       miou_arr[-1],
        'final_loss':       loss_arr[-1],
        'total_epochs':     len(miou_arr),
        'avg_epoch_s':      np.mean(history['time'])    if history.get('time')    else None,
        'total_train_min':  np.sum(history['time']) / 60 if history.get('time')   else None,
        'avg_gpu_gb':       np.mean(history['gpu_mem']) if history.get('gpu_mem') else None,
        'peak_gpu_gb':      np.max(history['gpu_mem'])  if history.get('gpu_mem') else None,
        'avg_lr_final3':    np.mean(history['lr'][-3:]) if history.get('lr')      else None,
    }


# ── inferência ────────────────────────────────────────────────────────────────

def build_model_from_ckpt(kind, model_size, cfg, device):
    from src.model import AttentionResUNet, Prithvi11BandsModel
    ds = cfg['dataset']
    nb = ds.get('num_bands', 11)
    nc = ds['num_classes']

    if kind == 'prithvi':
        model = Prithvi11BandsModel(nc, nb, pretrained=False, model_size=model_size)
    else:
        model = AttentionResUNet(nb, nc, model_size=model_size)

    ck = ckpt_path(cfg, kind)
    loaded = False
    if ck and os.path.isfile(ck):
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state.get('model_state_dict', state), strict=False)
        loaded = True
    return model.to(device), loaded


def count_params(model):
    t = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return t, tr


def file_mb(path):
    return os.path.getsize(path) / 1024 ** 2 if path and os.path.isfile(path) else None


@torch.no_grad()
def bench_inference(model, ds_cfg, batch_size, n_batches, device, is_prithvi):
    model.eval()
    nb = ds_cfg.get('num_bands', 11)
    cs = ds_cfg.get('crop_size', 512)
    x  = torch.randn(batch_size, nb, cs, cs, device=device)

    for _ in range(3):
        model(x.unsqueeze(2) if is_prithvi else x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        model(x.unsqueeze(2) if is_prithvi else x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ns = n_batches * batch_size
    return {
        'ms_batch':  elapsed / n_batches * 1000,
        'ms_sample': elapsed / ns * 1000,
        'sps':       ns / elapsed,
        'peak_gb':   torch.cuda.max_memory_allocated(device) / 1024 ** 3,
    }


# ── formatação ────────────────────────────────────────────────────────────────

W = 80

def fatline():  print('━' * W)
def thinline(): print('─' * W)

def section(title):
    print()
    fatline()
    print(f'  {title}')
    fatline()


def row(label, *vals, lw=32):
    cols = [f'{str(v):<14}' for v in vals]
    print(f'  {label:<{lw}}' + '  '.join(cols))


def fv(v, spec, fb='N/A'):
    return format(v, spec) if v is not None else fb


def na(v): return fv(v, '.4f')
def nb_(v): return fv(v, '.2f')
def ni(v): return str(v) if v is not None else 'N/A'


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"━"*W}')
    print(f'  BENCHMARK  —  Prithvi Tiny | Prithvi 100M | UNet R34 | UNet R50')
    fatline()
    print(f'  Dispositivo : {device}')
    if torch.cuda.is_available():
        print(f'  GPU         : {torch.cuda.get_device_name(device)}')
    print(f'  batch_size  : {args.batch_size}   n_batches inferência: {args.n_batches}')
    fatline()

    pcfg_base = load_yaml('config/prithvi.yaml')
    ucfg_base = load_yaml('config/unet.yaml')

    # monta configs de benchmark por variante
    variant_cfgs = {}
    for kind, size, name in VARIANTS:
        base = pcfg_base if kind == 'prithvi' else ucfg_base
        cfg, vdir = make_variant_config(kind, size, base)
        variant_cfgs[(kind, size)] = (cfg, vdir, name)

    # ── FASE 1: TREINO ────────────────────────────────────────────────────────
    if args.all:
        print()
        for kind, size, name in VARIANTS:
            cfg, vdir, _ = variant_cfgs[(kind, size)]
            ck = ckpt_path(cfg, kind)
            if not args.force and ck and os.path.isfile(ck):
                print(f'  [{name}] checkpoint já existe — pulando treino. (use --force para retreinar)')
                continue
            print(f'\n  ┌─ TREINANDO {name} ─────────────────────')
            t0 = time.time()
            run_training(kind, cfg)
            print(f'  └─ {name} concluído em {(time.time()-t0)/60:.1f} min')

    # ── FASE 2: COLETA DE RESULTADOS ──────────────────────────────────────────
    results = []
    for kind, size, name in VARIANTS:
        cfg, vdir, _ = variant_cfgs[(kind, size)]
        ck = ckpt_path(cfg, kind)

        print(f'\n  [{name}] carregando modelo e rodando inferência...')
        model, loaded = build_model_from_ckpt(kind, size, cfg, device)
        total, trainable = count_params(model)
        mb = file_mb(ck)

        inf = bench_inference(
            model, cfg['dataset'], args.batch_size, args.n_batches,
            device, is_prithvi=(kind == 'prithvi')
        )

        hist = load_history(ck)
        met  = extract_metrics(hist)

        results.append({
            'name': name, 'kind': kind, 'size': size,
            'total': total, 'trainable': trainable, 'mb': mb,
            'loaded': loaded, 'inf': inf, 'met': met,
        })

        del model
        torch.cuda.empty_cache()

    names = [r['name'] for r in results]

    # ── TABELA: MODELO ────────────────────────────────────────────────────────
    section('1 / 4  —  MODELO')
    row('', *names)
    thinline()
    row('Parâmetros totais (M)',     *[f"{r['total']/1e6:.1f}"     for r in results])
    row('Parâmetros treináveis (M)', *[f"{r['trainable']/1e6:.1f}" for r in results])
    row('Checkpoint .pth (MB)',      *[nb_(r['mb'])                for r in results])
    row('Checkpoint presente',       *['sim' if r['loaded'] else 'não' for r in results])

    # ── TABELA: INFERÊNCIA ────────────────────────────────────────────────────
    section(f'2 / 4  —  INFERÊNCIA  (batch={args.batch_size}  crop=512×512  n={args.n_batches})')
    row('', *names)
    thinline()
    row('ms / batch',          *[f"{r['inf']['ms_batch']:.1f}"  for r in results])
    row('ms / amostra',        *[f"{r['inf']['ms_sample']:.1f}" for r in results])
    row('amostras / segundo',  *[f"{r['inf']['sps']:.1f}"       for r in results])
    row('pico GPU — inf (GB)', *[f"{r['inf']['peak_gb']:.2f}"   for r in results])
    base_ms = results[0]['inf']['ms_batch']
    row(f'speedup vs {results[0]["name"]}',
        *[f"{base_ms/r['inf']['ms_batch']:.2f}×" for r in results])

    # ── TABELA: TREINO ────────────────────────────────────────────────────────
    section('3 / 4  —  TREINO  (métricas dos checkpoints)')
    row('', *names)
    thinline()
    row('Épocas treinadas',         *[ni(r['met'].get('total_epochs'))     for r in results])
    row('Tempo médio / época (s)',  *[fv(r['met'].get('avg_epoch_s'), '.1f') for r in results])
    row('Tempo total treino (min)', *[fv(r['met'].get('total_train_min'), '.1f') for r in results])
    row('Mem GPU média treino (GB)',*[nb_(r['met'].get('avg_gpu_gb'))      for r in results])
    row('Mem GPU pico treino (GB)', *[nb_(r['met'].get('peak_gpu_gb'))     for r in results])
    row('LR final (média 3 ep.)',   *[fv(r['met'].get('avg_lr_final3'), '.2e') for r in results])

    # ── TABELA: QUALIDADE ─────────────────────────────────────────────────────
    section('4 / 4  —  QUALIDADE  (validação)')
    row('', *names)
    thinline()
    row('Best mIoU',                *[na(r['met'].get('best_miou'))       for r in results])
    row('  → época',                *[ni(r['met'].get('best_miou_epoch')) for r in results])
    row('Best pixel accuracy',      *[na(r['met'].get('best_acc'))        for r in results])
    row('Best val loss',            *[na(r['met'].get('best_val_loss'))   for r in results])
    row('  → época',                *[ni(r['met'].get('best_loss_epoch')) for r in results])
    row('mIoU final (last epoch)',  *[na(r['met'].get('final_miou'))      for r in results])
    row('Loss final (last epoch)',  *[na(r['met'].get('final_loss'))      for r in results])

    print()
    fatline()
    print(f'  Resultados salvos em: {os.path.abspath(BENCH_DIR)}')
    fatline()
    print()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Benchmark completo de segmentação')
    p.add_argument('--all',        action='store_true',
                   help='Treina todas as variantes antes de comparar')
    p.add_argument('--force',      action='store_true',
                   help='Re-treina mesmo se checkpoint já existir (requer --all)')
    p.add_argument('--batch-size', type=int, default=4,
                   help='Batch size para benchmark de inferência')
    p.add_argument('--n-batches',  type=int, default=20,
                   help='Número de batches para benchmark de inferência')
    main(p.parse_args())
