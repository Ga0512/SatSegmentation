import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from torch.utils.data import Dataset
import kornia.augmentation as K
gdal.UseExceptions()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SegDatasetMemmap(Dataset):
    def __init__(self, img_memmap_path, mask_memmap_path, file_list, pmins, pmaxs, means, stds, num_bands, crop_size):

        self.img_path = img_memmap_path
        self.mask_path = mask_memmap_path
        self.N = len(file_list)
        self.num_bands = num_bands
        self.crop_size = crop_size
        self.imgs = None
        self.masks = None
        # Estatísticas expostas para uso no loop de treino (normalização na GPU)
        self.pmins = torch.tensor(pmins, dtype=torch.float32)[:, None, None]
        self.pmaxs = torch.tensor(pmaxs, dtype=torch.float32)[:, None, None]
        self.means = torch.tensor(means, dtype=torch.float32)[:, None, None]
        self.stds  = torch.tensor(stds,  dtype=torch.float32)[:, None, None]

        logging.info(f'Dataset consistente: {self.N} amostras | {crop_size}x{crop_size}')

    def _lazy_init(self):
        if self.imgs is None:
            self.imgs = np.memmap(self.img_path, dtype='uint16', mode='r', shape=(self.N, self.num_bands, self.crop_size, self.crop_size))
            self.masks = np.memmap(self.mask_path, dtype='int16', mode='r', shape=(self.N, self.crop_size, self.crop_size))

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        self._lazy_init()

        img  = torch.as_tensor(np.array(self.imgs[idx]),  dtype=torch.float32)
        mask = torch.as_tensor(np.array(self.masks[idx]), dtype=torch.long)

        return img, mask


class PrithviDataset(Dataset):
    def __init__(self, img_p, mask_p, num_samples, means, stds, num_bands, crop_size):
        self.img_p, self.mask_p = img_p, mask_p
        self.N = num_samples
        # Convertemos para tensor aqui, mas eles ficam na RAM por enquanto
        self.means = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.stds = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1)
        self.imgs = None
        self.num_bands = num_bands
        self.crop_size = crop_size

    def __len__(self): return self.N
    
    def __getitem__(self, i):
        if self.imgs is None:
            self.imgs = np.memmap(self.img_p, dtype='uint16', mode='r', shape=(self.N, self.num_bands, self.crop_size, self.crop_size))
            self.masks = np.memmap(self.mask_p, dtype='int16', mode='r', shape=(self.N, self.crop_size, self.crop_size))
        
        img  = torch.as_tensor(np.array(self.imgs[i]),  dtype=torch.float32)
        mask = torch.as_tensor(np.array(self.masks[i]), dtype=torch.long)
        
        return img, mask


# 3. AUGMENTAÇÃO - GPU
def build_gpu_augmenter():
    geometric = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation90(times=(1, 3), p=0.5),
        K.RandomBrightness(brightness=(0.9, 1.1), p=0.5),
        K.RandomContrast(contrast=(0.9, 1.1), p=0.5),
        K.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1), align_corners=True),
        data_keys=['input', 'mask'],
        same_on_batch=False)

    photometric = K.AugmentationSequential(K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.3), data_keys=['input'])

    return geometric.to(device), photometric.to(device)

def match_mask_file(img_fname, labels_dir):
    base_name = os.path.splitext(img_fname)[0]
    for f in os.listdir(labels_dir):
        if base_name in f and f.lower().endswith('.tif'):
            return f
    return None

def build_memmap_and_stats(images_dir, labels_dir, memmap_dir, file_list, num_bands, img_size, crop_size=512,
                           split_name='train', p_low=2, p_high=98, sample_per_image=10000, compute_stats=True):

    H, W = img_size
    crops_per_row = H // crop_size
    crops_per_col = W // crop_size
    crops_per_image = crops_per_row * crops_per_col
    total_samples = len(file_list) * crops_per_image

    img_path   = os.path.join(memmap_dir, f'{split_name}_images.dat')
    mask_path  = os.path.join(memmap_dir, f'{split_name}_masks.dat')
    list_path  = os.path.join(memmap_dir, f'{split_name}_files.npy')
    stats_path = os.path.join(memmap_dir, f'{split_name}_stats.npz')

    if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(stats_path):
        logging.info(f'[memmap] {split_name}: carregando dados existentes.')
        stats = np.load(stats_path)
        return (img_path, mask_path, np.load(list_path, allow_pickle=True), stats['pmins'], stats['pmaxs'], stats['means'], stats['stds'])

    logging.info(f'[memmap] Construindo {split_name} (multi-band read)...')

    imgs_mm = np.memmap(img_path, dtype='uint16', mode='w+', shape=(total_samples, num_bands, crop_size, crop_size))

    masks_mm = np.memmap(mask_path, dtype='int16', mode='w+', shape=(total_samples, crop_size, crop_size))

    # --- acumuladores estatísticos ---
    if compute_stats:
        band_samples = [[] for _ in range(num_bands)]
        sums    = np.zeros(num_bands, dtype=np.float64)
        sq_sums = np.zeros(num_bands, dtype=np.float64)
        counts  = np.zeros(num_bands, dtype=np.float64)

    names = []
    sample_idx = 0

    for fname in tqdm(file_list, desc=f'Processando {split_name}'):
        base = os.path.splitext(fname)[0]

        ds_img  = gdal.Open(os.path.join(images_dir, base + '.tif'), gdal.GA_ReadOnly)
        ds_mask = gdal.Open(os.path.join(labels_dir, base + '_mask.tif'), gdal.GA_ReadOnly)

        # Lê a imagem inteira
        full_img = ds_img.ReadAsArray()  # shape: (bands, H, W)
        full_mask = ds_mask.GetRasterBand(1).ReadAsArray()

        # Trata NoData das bandas
        for b in range(num_bands):
            band = ds_img.GetRasterBand(b + 1)
            nodata = band.GetNoDataValue()
            if nodata is not None:
                full_img[b][full_img[b] == nodata] = 0

        # Trata NoData da máscara
        nodata_mask = ds_mask.GetRasterBand(1).GetNoDataValue()
        if nodata_mask is not None:
            full_mask[full_mask == nodata_mask] = 0

        # --- slicing em memória ---
        for row in range(crops_per_row):
            for col in range(crops_per_col):

                y0 = row * crop_size
                x0 = col * crop_size

                img_block = full_img[:, y0:y0+crop_size, x0:x0+crop_size]
                mask_block = full_mask[y0:y0+crop_size, x0:x0+crop_size]

                # Estatísticas por banda
                if compute_stats:
                    for b in range(num_bands):
                        arr = img_block[b]
                        valid = arr[arr > 0].astype(np.float64)

                        if valid.size > 0:
                            if valid.size > sample_per_image:
                                idx = np.random.choice(valid.size, sample_per_image, replace=False)
                                valid_sample = valid[idx]
                            else:
                                valid_sample = valid

                            band_samples[b].append(valid_sample)
                            sums[b]    += valid.sum()
                            sq_sums[b] += (valid ** 2).sum()
                            counts[b]  += valid.size

                imgs_mm[sample_idx]  = img_block.astype(np.uint16)
                masks_mm[sample_idx] = mask_block.astype(np.int16)

                names.append(f'{base}_r{row}_c{col}')
                sample_idx += 1

        ds_img = None
        ds_mask = None

    imgs_mm.flush()
    masks_mm.flush()

    # ===== finaliza estatísticas =====
    if compute_stats:
        pmins = np.zeros(num_bands, dtype=np.float32)
        pmaxs = np.zeros(num_bands, dtype=np.float32)

        for b in range(num_bands):
            concat = np.concatenate(band_samples[b])
            pmins[b] = np.percentile(concat, p_low)
            pmaxs[b] = np.percentile(concat, p_high)

        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts>0)
        var   = np.divide(sq_sums, counts, out=np.zeros_like(sq_sums), where=counts>0) - means**2
        stds  = np.sqrt(np.clip(var, 1e-12, None))

        np.save(list_path, np.array(names))
        np.savez(stats_path, pmins=pmins, pmaxs=pmaxs, means=means, stds=stds)

    else:
        pmins = pmaxs = means = stds = None
        np.save(list_path, np.array(names))

    logging.info(f'[memmap] {split_name} pronto. Total samples: {total_samples}')

    return img_path, mask_path, np.array(names), pmins, pmaxs, means, stds

def build_memmap_and_stats_prithvi(images_dir, labels_dir, memmap_dir, file_list, num_bands, img_size, crop_size):
    H, W = img_size
    crops_per_row, crops_per_col = H // crop_size, W // crop_size
    total_samples = len(file_list) * crops_per_row * crops_per_col
    img_path = os.path.join(memmap_dir, 'images.dat')
    mask_path = os.path.join(memmap_dir, 'masks.dat')
    stats_path = os.path.join(memmap_dir, 'stats.npz')
    if os.path.exists(stats_path):
        s = np.load(stats_path)
        return img_path, mask_path, s['means'], s['stds']
    imgs_mm = np.memmap(img_path, dtype='uint16', mode='w+', shape=(total_samples, num_bands, crop_size, crop_size))
    masks_mm = np.memmap(mask_path, dtype='int16', mode='w+', shape=(total_samples, crop_size, crop_size))
    sums, sq_sums, counts = np.zeros(num_bands), np.zeros(num_bands), 0
    idx = 0
    for fname in tqdm(file_list, desc='Processando TIFs para Memmap'):
        mask_fname = match_mask_file(fname, labels_dir)
        if not mask_fname: continue
        img_ds = gdal.Open(os.path.join(images_dir, fname))
        mask_ds = gdal.Open(os.path.join(labels_dir, mask_fname))
        img_arr = img_ds.ReadAsArray()
        mask_arr = mask_ds.ReadAsArray()
        for r in range(crops_per_row):
            for c in range(crops_per_col):
                y, x = r*crop_size, c*crop_size
                imgs_mm[idx] = img_arr[:, y:y+crop_size, x:x+crop_size]
                masks_mm[idx] = mask_arr[y:y+crop_size, x:x+crop_size]
                for b in range(num_bands):
                    valid = imgs_mm[idx][b].astype(np.float64)
                    sums[b] += valid.sum()
                    sq_sums[b] += (valid**2).sum()
                counts += (crop_size * crop_size)
                idx += 1
    means = sums / counts
    stds = np.sqrt(np.clip((sq_sums / counts) - means**2, 1e-10, None))
    np.savez(stats_path, means=means, stds=stds)
    imgs_mm.flush()
    return img_path, mask_path, means, stds
