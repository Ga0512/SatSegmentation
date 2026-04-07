import os
os.environ["PROJ_DATA"] = r"..\venv\Lib\site-packages\rasterio\proj_data"
import time
import torch
import numpy as np
from osgeo import gdal
import torch.nn.functional as F
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 19
NUM_BANDS = 11
CROP_SIZE = 512
BATCH_SIZE = 32
MODEL_PATH = './model/prithvi_production_fp16.pt'
INPUT_DIR = '../new/data/Images'
OUTPUT_DIR = './predicoes'

stats = np.load('./memmap_output/stats.npz')
means = torch.tensor(stats['means'], dtype=torch.float32).view(1, -1, 1, 1, 1).to(DEVICE)
stds = torch.tensor(stats['stds'], dtype=torch.float32).view(1, -1, 1, 1, 1).to(DEVICE)


def run_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = torch.jit.load(MODEL_PATH, map_location='cuda')
    model.half().eval()

    tifs = sorted(Path(INPUT_DIR).glob('*.tif'))
    total = len(tifs)
    print(f"Encontradas {total} imagens\n")

    # --- Fase 1: coletar todos os crops de todas as imagens ---
    image_meta = []  # (path, H, W, geo_transform, projection)
    all_crops = []   # np arrays
    all_info = []    # (img_idx, y, x, cy, cx)

    for img_idx, tif_path in enumerate(tifs):
        ds = gdal.Open(str(tif_path))
        img_full = ds.ReadAsArray()
        _, H, W = img_full.shape
        image_meta.append((tif_path, H, W, ds.GetGeoTransform(), ds.GetProjection()))
        ds = None

        for y in range(0, H, CROP_SIZE):
            for x in range(0, W, CROP_SIZE):
                crop = img_full[:, y:y+CROP_SIZE, x:x+CROP_SIZE].astype(np.float32)
                ch, cy, cx = crop.shape
                if cy < CROP_SIZE or cx < CROP_SIZE:
                    padded = np.zeros((ch, CROP_SIZE, CROP_SIZE), dtype=np.float32)
                    padded[:, :cy, :cx] = crop
                    crop = padded
                all_crops.append(crop)
                all_info.append((img_idx, y, x, cy, cx))

    print(f"Total de crops: {len(all_crops)} de {total} imagens — batch_size={BATCH_SIZE}\n")

    # prediction maps por imagem
    pred_maps = [np.zeros((m[1], m[2]), dtype=np.uint8) for m in image_meta]
    img_timers = [0.0] * total

    # --- Fase 2: inferência em batches cross-image ---
    total_start = time.time()

    with torch.no_grad():
        for b in range(0, len(all_crops), BATCH_SIZE):
            batch_crops = all_crops[b:b+BATCH_SIZE]
            batch_info = all_info[b:b+BATCH_SIZE]

            batch_tensor = torch.from_numpy(np.stack(batch_crops))
            batch_tensor = batch_tensor.unsqueeze(2).to(DEVICE)
            batch_tensor = ((batch_tensor - means) / (stds + 1e-6)).half()

            t0 = time.time()
            logits = model(batch_tensor)
            dt = time.time() - t0

            if logits.shape[-2:] != (CROP_SIZE, CROP_SIZE):
                logits = F.interpolate(logits, size=(CROP_SIZE, CROP_SIZE), mode='bilinear')

            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

            for i, (img_idx, y, x, cy, cx) in enumerate(batch_info):
                pred_maps[img_idx][y:y+cy, x:x+cx] = preds[i, :cy, :cx]
                img_timers[img_idx] += dt / len(batch_info)

    # --- Fase 3: salvar cada imagem ---
    for img_idx, (tif_path, H, W, geo, proj) in enumerate(image_meta):
        out_path = os.path.join(OUTPUT_DIR, tif_path.stem + '_pred.tif')
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(out_path, W, H, 1, gdal.GDT_Byte)
        out_ds.SetGeoTransform(geo)
        out_ds.SetProjection(proj)
        out_ds.GetRasterBand(1).WriteArray(pred_maps[img_idx])
        out_ds.FlushCache()
        out_ds = None
        print(f"[{img_idx+1}/{total}] {tif_path.name} — {H}x{W} — ~{img_timers[img_idx]:.2f}s GPU")

    total_time = time.time() - total_start
    print(f"\nConcluído! {total} imagens, {len(all_crops)} crops em {total_time:.1f}s ({total_time/total:.1f}s/img)")


if __name__ == "__main__":
    run_inference()