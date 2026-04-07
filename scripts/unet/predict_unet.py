import torch
import logging
import numpy as np
import onnxruntime as ort
import rasterio
from tqdm import tqdm
from pathlib import Path

# PATHS E PARÂMETROS
onnx_path = './model/unet_sf4_fp16.onnx'
stats_path = './output/memmap/train_stats.npz'
input_dir = '../new/data/Images'
output_dir = './Masks'
log_file = 'SF4_Unet_log_inferencia.txt'

num_bands = 11
num_classes = 19
tile_size = 512
overlap = 128
batch_size = 32

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Path(output_dir).mkdir(parents=True, exist_ok=True)

def build_weight_map_tensor(size, device):
    ax = torch.linspace(-1.0, 1.0, size, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    dist = torch.sqrt(xx**2 + yy**2)
    weights = torch.exp(-(dist**2) / (2 * 0.5**2))
    weights = weights / weights.max()
    return weights.unsqueeze(0).unsqueeze(0)

weight_map = build_weight_map_tensor(tile_size, device)

stats = np.load(stats_path)
pmins = torch.tensor(stats['pmins'], dtype=torch.float32, device=device).view(1, num_bands, 1, 1)
pmaxs = torch.tensor(stats['pmaxs'], dtype=torch.float32, device=device).view(1, num_bands, 1, 1)
means = torch.tensor(stats['means'], dtype=torch.float32, device=device).view(1, num_bands, 1, 1)
stds  = torch.tensor(stats['stds'],  dtype=torch.float32, device=device).view(1, num_bands, 1, 1)

def normalize_batch(img_batch_np):
    img_t = torch.from_numpy(img_batch_np).float().to(device)
    img_t = torch.clamp(img_t, pmins, pmaxs)
    img_t = (img_t - means) / (stds + 1e-6)
    return img_t.cpu().numpy().astype(np.float16)

logging.info('Carregando modelo ONNX Runtime...')
providers = [('CUDAExecutionProvider', {})]
ort_session = ort.InferenceSession(onnx_path, providers=providers)
input_name = ort_session.get_inputs()[0].name
logging.info('Modelo carregado.')

images = sorted(Path(input_dir).glob('*.tif'))
images = [p for p in images if not (Path(output_dir) / f'{p.stem}_mask.tif').exists()]
logging.info(f'{len(images)} imagens pra processar')

accumulators = {}
patches, metas = [], []

def flush_batch():
    if not patches: return
    batch_norm = normalize_batch(np.stack(patches))
    logits_np = ort_session.run(None, {input_name: batch_norm})[0]

    logits_t = torch.tensor(logits_np, device=device)
    probs_t = torch.softmax(logits_t, dim=1)
    probs_weighted = (probs_t * weight_map).cpu().to(torch.float16)
    weights_cpu = weight_map[0].cpu().to(torch.float16)

    for i, (img_idx, cx, cy) in enumerate(metas):
        prob, weight, _, _ = accumulators[img_idx]
        h_end = min(tile_size, prob.shape[1] - cy)
        w_end = min(tile_size, prob.shape[2] - cx)
        prob[:, cy:cy+h_end, cx:cx+w_end] += probs_weighted[i, :, :h_end, :w_end]
        weight[:, cy:cy+h_end, cx:cx+w_end] += weights_cpu[:, :h_end, :w_end]

    patches.clear()
    metas.clear()

def add_tile(img_data, img_idx, x, y):
    tile = img_data[:, y:y+tile_size, x:x+tile_size]
    patches.append(tile)
    metas.append((img_idx, x, y))
    if len(patches) == batch_size:
        flush_batch()

step = tile_size - overlap

for img_idx, input_image in enumerate(tqdm(images, desc="Imagens")):
    with rasterio.open(input_image) as src:
        W, H = src.width, src.height
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw',
                       tiled=True, blockxsize=256, blockysize=256, bigtiff='yes')
        img_data = src.read()

    output_path = Path(output_dir) / f'{input_image.stem}_mask.tif'
    accumulators[img_idx] = (
        torch.zeros((num_classes, H, W), dtype=torch.float16),
        torch.zeros((1, H, W), dtype=torch.float16),
        profile,
        output_path
    )

    # Grid principal
    for y in range(0, H - tile_size + 1, step):
        for x in range(0, W - tile_size + 1, step):
            add_tile(img_data, img_idx, x, y)

    # Borda direita
    if (W - tile_size) % step != 0:
        for y in range(0, H - tile_size + 1, step):
            add_tile(img_data, img_idx, W - tile_size, y)

    # Borda inferior
    if (H - tile_size) % step != 0:
        for x in range(0, W - tile_size + 1, step):
            add_tile(img_data, img_idx, x, H - tile_size)

    # Canto inferior direito
    if (W - tile_size) % step != 0 and (H - tile_size) % step != 0:
        add_tile(img_data, img_idx, W - tile_size, H - tile_size)

    del img_data

flush_batch()

for img_idx, (prob, weight, profile, output_path) in accumulators.items():
    weight[weight == 0] = 1e-6
    prob /= weight
    pred = torch.argmax(prob, dim=0).byte().numpy()

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pred, 1)

    logging.info(f'Salvo: {output_path}')

del accumulators
logging.info(f'Inferência finalizada. {len(images)} imagens processadas.')
