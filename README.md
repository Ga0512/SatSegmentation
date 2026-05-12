# Satellite Segmentation Framework

Framework de segmentação semântica para imagens de satélite multiespectrais, comparando duas arquiteturas:

- **Prithvi**: Foundation model geoespacial IBM/NASA (ViT pré-treinado em dados HLS — Landsat/Sentinel-2 harmonizados). Variantes `tiny`, `100m`, `300m`, `600m`
- **AttentionResUNet**: Encoder ResNet34/50 + decoder com attention gates espaciais e de canal

O framework é **agnóstico a sensor e número de classes**: a quantidade de bandas, a ordem delas e o número de classes são definidos via YAML. O mapeamento das bandas de entrada para os pesos pré-treinados do HLS é explícito (`model_bands`), o que evita inicializar com peso errado quando a ordem das bandas difere do HLS.

![alt text](asset/image.png)

---

## Sumário

1. [Arquitetura do Projeto](#arquitetura-do-projeto)
2. [Setup — Windows (venv)](#setup--windows-venv)
3. [Setup — Docker](#setup--docker)
4. [Pipeline de Dados](#pipeline-de-dados)
5. [Mapeamento de Bandas](#mapeamento-de-bandas)
6. [Treinamento](#treinamento)
7. [Inferência](#inferência)
8. [Exportação de Modelos](#exportação-de-modelos)
9. [Estrutura de Diretórios](#estrutura-de-diretórios)
10. [Configurações](#configurações)
11. [Métricas e Monitoramento](#métricas-e-monitoramento)
12. [Benchmark](#benchmark)

---

## Arquitetura do Projeto

```
SatSegmentation/
├── config/
│   ├── prithvi.yaml          # Hiperparâmetros do Prithvi (incluindo model_bands)
│   └── unet.yaml             # Hiperparâmetros do U-Net
├── scripts/
│   ├── prithvi/
│   │   ├── train_prithvi.py  # Loop de treino Prithvi
│   │   ├── predict_prithvi.py# Inferência em batch (TorchScript FP16)
│   │   └── prithvi_fp16.py   # Conversão para FP16 + TorchScript
│   └── unet/
│       ├── train_unet.py     # Loop de treino U-Net
│       ├── predict_unet.py   # Inferência ONNX com tiling e overlap
│       └── export_onnx_unet.py
├── src/
│   ├── model.py              # AttentionResUNet + Prithvi11BandsModel (com _HLS_BAND_INDEX)
│   ├── dataset.py            # SegDatasetMemmap, PrithviDataset, augmentação GPU
│   ├── metrics.py            # FocalDiceLoss, mIoU, Dice, Kappa, plots
│   ├── utils.py              # Pesos de classe, avaliação, file pairing
│   ├── checkpoint_model.py   # Save/load de estado completo de treino
│   ├── eval.py               # Loop de validação com CSV + confusion matrix
│   └── fix_terratorch.py     # Patch de instalação do TerraTorch (one-time)
├── data/
│   ├── img/                  # GeoTIFFs de entrada
│   └── mask/                 # Máscaras de segmentação (*_mask.tif)
└── memmap_output/            # Dados pré-processados (gerado automaticamente)
```

---

## Setup — Windows (venv)

```powershell
git clone https://github.com/Ga0512/SatSegmentation.git
cd SatSegmentation

python -m venv venv
venv\Scripts\activate

python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Patch obrigatório (executar uma única vez após instalar):**
```powershell
python -m src.fix_terratorch
```

**Variável de ambiente para rasterio (necessária em toda sessão):**
```powershell
$env:PROJ_LIB = "$pwd\venv\Lib\site-packages\rasterio\proj_data"
```

---

## Setup — Docker

### Pré-requisitos

- Docker
- NVIDIA Container Toolkit (para acesso à GPU dentro do container)

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verificar acesso à GPU
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Build e execução

```bash
docker build -t satseg .
docker run -it --gpus all -v ${PWD}:/app satseg
```

Dentro do container:
```bash
python -m src.fix_terratorch
```

---

## Pipeline de Dados

O pipeline transforma GeoTIFFs brutos em arquivos memory-mapped binários (`.dat`) para carregamento eficiente durante o treino.

### Fluxo

```
GeoTIFFs (./data/) → build_memmap_and_stats() → ./memmap_output/*.dat + stats.npz
                                                              ↓
                                               SegDatasetMemmap / PrithviDataset
                                                              ↓
                                               DataLoader (workers CPU, pin_memory)
                                                              ↓
                                               Normalização na GPU (clamp + z-score)
                                                              ↓
                                               Augmentação GPU (Kornia)
```

### Detalhes técnicos

- Imagens armazenadas como `uint16` (2 bytes/pixel), máscaras como `int16`
- Crops não-sobrepostos do tamanho `crop_size` extraídos das imagens originais
- Estatísticas por banda (percentis p2/p98, média, desvio padrão) calculadas uma única vez e salvas em `stats.npz`
- **Normalização executada na GPU** (clamp percentílico + z-score por banda) — os workers de CPU fazem apenas leitura e cast de tipo
- A construção do memmap é automática na primeira execução; execuções subsequentes reutilizam os arquivos existentes

### Tratamento de NoData e índices inválidos

A máscara passa por duas defesas durante a construção do memmap:

1. **NoData**: pixels com o valor `NoDataValue` declarado nos metadados do GeoTIFF (da imagem **ou** da máscara) são marcados como `IGNORE_INDEX = 255`. Esses pixels são ignorados pela loss (`ignore_index=255`) e nas métricas.
2. **Índices fora de `[0, num_classes)`**: quando `num_classes` é passado ao builder, qualquer valor de classe residual (ex: anotações antigas com classe 4 em um setup atual de 3 classes) é convertido em `IGNORE_INDEX` e logado. Evita crash do `F.one_hot` no `FocalDiceLoss` (CUDA device-side assert) e o silencioso enviesamento de métricas.

`class 0` é **classe válida (background)**, não NoData — é tratada como qualquer outra classe pela loss e pelas métricas.

---

## Mapeamento de Bandas

O Prithvi foi pré-treinado em 6 bandas HLS, nesta ordem:

| Posição HLS | Banda       |
|-------------|-------------|
| 0           | BLUE        |
| 1           | GREEN       |
| 2           | RED         |
| 3           | NIR_NARROW  |
| 4           | SWIR_1      |
| 5           | SWIR_2      |

No YAML, o campo `dataset.model_bands` declara **qual banda HLS cada posição da sua entrada representa**. O `Prithvi11BandsModel` lê isso e copia o peso pré-treinado correto para cada canal de `patch_embed.proj`, em vez de assumir cegamente a ordem `[BLUE, GREEN, RED, ...]`.

### Exemplos

**Landsat 8/9 — B4, B5, B6, B7** (canônico para exploração mineral, clay/silica via SWIR):

```yaml
dataset:
  num_bands: 4
  model_bands: ['RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']
```

**Sentinel-2 — B2, B3, B4, B8A, B11, B12**:

```yaml
dataset:
  num_bands: 6
  model_bands: ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']
```

### Comportamento de fallback

- Se `model_bands` **não** for declarado: o modelo cai no comportamento antigo — copia as primeiras `N` bandas do HLS na ordem default. Bandas extras (além de 6) são inicializadas com a média dos pesos HLS.
- Se um nome em `model_bands` **não** existir em `_HLS_BAND_INDEX` (ex: `RED_EDGE`): hoje o modelo levanta `ValueError`. Para sensores com bandas fora do HLS (Red Edge do Sentinel-2, Coastal Aerosol), seria preciso estender `_HLS_BAND_INDEX` com uma estratégia de inicialização para a banda nova.

> **Trocar de sensor** = mudar `num_bands` + `model_bands` no YAML, apagar `./memmap_output/`, e rodar. Sem mexer em código.

---

## Treinamento

### Prithvi

```bash
python -m scripts.prithvi.train_prithvi --config config/prithvi.yaml

# Retomar de checkpoint
python -m scripts.prithvi.train_prithvi --config config/prithvi.yaml --resume
python -m scripts.prithvi.train_prithvi --config config/prithvi.yaml --resume path/to/checkpoint.pth
```

**Características do loop:**
- Mixed precision (AMP FP16) com gradient clipping (`max_norm=1.0`)
- Scheduler: Linear warmup + Cosine Annealing
- Loss: **`FocalDiceLoss`** com `class_weights`, `focal_gamma`, `focal_weight`, `dice_weight`, `ignore_index=255`
- LR diferenciado: `learning_rate` para o backbone pré-treinado e `decoder_lr` (≥ backbone LR) para o decoder/head
- `oversample_rare_classes: true` — usa `WeightedRandomSampler` para visitar mais frequentemente patches com classes minoritárias
- Métricas: `JaccardIndex` (mIoU) e `Accuracy` via TorchMetrics

### AttentionResUNet

```bash
python -m scripts.unet.train_unet --config config/unet.yaml
```

**Características do loop:**
- Mixed precision (AMP FP16) com gradient accumulation (effective batch = `batch_size × grad_accum`)
- Scheduler: Cosine Annealing
- Loss: `FocalDiceLoss` com `class_weights`, `ignore_index=255`
- Augmentação GPU via Kornia: flip, rotação 90°, brilho/contraste, affine, blur gaussiano
- `torch.channels_last` no modelo para melhor throughput em GPUs com Tensor Cores

### Pesos de classe

Calculados automaticamente por `compute_class_weights()`:
- Frequência inversa com suavização logarítmica
- Clipping para evitar pesos extremos
- Pixels marcados como `IGNORE_INDEX` (NoData ou out-of-range) são excluídos da contagem
- Normalizados para média 1.0 sobre as classes ativas

---

## Inferência

### Prithvi (TorchScript FP16)

```bash
python -m scripts.prithvi.predict_prithvi
```

- Carrega modelo TorchScript em FP16 de `./model/prithvi_production_fp16.pt`
- Processa múltiplas imagens em batch cross-image (sem recarregar o modelo entre arquivos)
- Normalização z-score por banda na GPU antes da inferência
- Saída: GeoTIFF com máscara de classes em `./predicoes/`

### AttentionResUNet (ONNX Runtime)

```bash
python -m scripts.unet.predict_unet
```

- Inferência via ONNX Runtime com `CUDAExecutionProvider`
- Tiling com overlap e média ponderada por mapa gaussiano centrado (elimina artefatos de borda)
- Normalização clamp + z-score na GPU antes de cada batch
- Saída: GeoTIFF comprimido (LZW, tiled) em `./Masks/`

---

## Exportação de Modelos

### Prithvi → FP16 TorchScript

```bash
python -m scripts.prithvi.prithvi_fp16
```

Gera dois artefatos:
- `./model/best_prithvi_*_fp16.pth` — state dict em FP16
- `./model/prithvi_production_fp16.pt` — TorchScript standalone (sem dependência de `src/`)

### AttentionResUNet → ONNX

```bash
python -m scripts.unet.export_onnx_unet
```

---

## Estrutura de Diretórios

| Diretório | Conteúdo |
|---|---|
| `./data/img/` | GeoTIFFs de entrada (N bandas, dimensões definidas em `img_size`) |
| `./data/mask/` | Máscaras `*_mask.tif` (valores em `[0, num_classes)` ou NoData) |
| `./memmap_output/` | `*.dat` + `stats.npz` (gerado automaticamente) |
| `./model/` | Pesos Prithvi (`.pth`, FP16 `.pt`) |
| `./output/` | Melhor checkpoint U-Net, CSV de validação, confusion matrix |
| `./metrics_prithvi/` | Curvas de treino do Prithvi (PNG) |
| `./metrics_unet/` | Curvas de treino do U-Net (PNG) |
| `./predicoes/` | GeoTIFFs de saída da inferência Prithvi |
| `./Masks/` | GeoTIFFs de saída da inferência U-Net |

---

## Configurações

Os dois configs são totalmente genéricos: `num_bands`, `num_classes`, `img_size`, `crop_size`, hiperparâmetros e arquitetura são definidos no YAML. O código não tem nenhum valor hard-coded de classe ou banda.

### `config/prithvi.yaml`

```yaml
paths:
  images_dir: "<dir das imagens>"
  labels_dir: "<dir das máscaras>"
  memmap_dir:  "<dir do memmap>"
  save_model_path: "<caminho do checkpoint>"

dataset:
  img_size: [<H>, <W>]
  crop_size: <int>
  num_bands:   <N>           # qualquer N >= 1
  num_classes: <K>           # qualquer K >= 2 (inclui background se houver)
  val_size: <float>          # fração de validação
  num_cores: <int>           # workers do DataLoader
  # Opcional: mapeia cada posição da entrada para a banda HLS correspondente
  # (ver seção "Mapeamento de Bandas"). Se omitido, usa fallback.
  model_bands: ['<HLS_BAND>', ...]   # len(model_bands) == num_bands

training:
  epochs:     <int>
  patience:   <int>          # early stopping
  learning_rate: <float>     # LR do backbone pré-treinado
  decoder_lr:    <float>     # LR do decoder/head (geralmente ≥ backbone LR)
  batch_size:    <int>
  weight_decay:  <float>
  focal_gamma:   <float>     # γ da Focal — foco em pixels difíceis
  focal_weight:  <float>     # peso do termo Focal na loss
  dice_weight:   <float>     # peso do termo Dice na loss
  oversample_rare_classes: <bool>    # WeightedRandomSampler

model_size: <tiny | 100m | 300m | 600m>
```

### `config/unet.yaml`

```yaml
paths:
  images_dir: "<dir das imagens>"
  labels_dir: "<dir das máscaras>"
  csv_validation:  "<csv de validação>"
  checkpoint_path: "<dir de saída>"
  best_model_path: "<caminho do melhor modelo>"
  memmap_dir:      "<dir do memmap>"

dataset:
  img_size: [<H>, <W>]
  crop_size: <int>
  num_bands:   <N>
  num_classes: <K>
  val_size:  <float>
  num_cores: <int>

training:
  epochs:        <int>
  patience:      <int>
  learning_rate: <float>
  batch_size:    <int>
  grad_accum:    <int>       # effective batch = batch_size × grad_accum

model_size: <resnet34 | resnet50>
```

---

## Métricas e Monitoramento

A cada época são registrados e plotados:

| Métrica | Descrição |
|---|---|
| `train_loss` / `val_loss` | Loss média por época |
| `val_miou` | mIoU médio das classes de foreground (exclui background da média final) |
| `val_acc` | Pixel accuracy |
| `lr` | Learning rate atual |
| `time` | Tempo de execução por época (s) |
| `gpu_mem` | Pico de memória GPU alocada (GB) |

Gráficos salvos em `./metrics_prithvi/` e `./metrics_unet/` após cada época.

Ao final do treino, `evaluate_model()` gera:
- CSV com precision, recall, F1, IoU e Dice por classe
- Confusion matrix em PNG
- Kappa de Cohen e Weighted IoU globais

### Notas de implementação

- **`src/fix_terratorch.py`** deve ser executado uma vez após a instalação — corrige a constante `SENTINEL2_ALL_SOFTCON → SENTINEL2_ALL_MOCO` na biblioteca TerraTorch instalada
- Todos os scripts são invocados como módulos (`python -m scripts.X.Y`), não diretamente
- O Prithvi adapta o `patch_embed.proj` original (6 canais HLS) para o número de bandas declarado em `num_bands`, copiando o peso pré-treinado de cada banda HLS correta para a posição certa via `model_bands`
- `IGNORE_INDEX = 255` em `src/dataset.py` é o sentinela para NoData e índices inválidos — mantido consistente entre o builder de memmap, a `FocalDiceLoss` e as métricas
- Não há testes unitários; a avaliação é integrada ao loop de treino e produz CSVs e plots automaticamente

---

## Benchmark

> Os números abaixo foram coletados em uma configuração específica (11 bandas, 19 classes). Servem como referência **relativa** entre arquiteturas — valores absolutos variam com o número de bandas, classes, tamanho da imagem e qualidade do dataset.

**Dispositivo:** cuda  
**GPU:** NVIDIA GeForce RTX 3060 Laptop GPU  
**batch_size:** 4 | **n_batches inferência:** 20

### 1/4 — MODELO

| Métrica                      | Prithvi-Tiny | Prithvi-100M | UNet-R34 | UNet-R50 |
|------------------------------|--------------|--------------|----------|----------|
| Parâmetros totais (M)        | 13.2         | 96.9         | 24.7     | 75.5     |
| Parâmetros treináveis (M)    | 13.2         | 96.9         | 24.7     | 75.5     |
| Checkpoint .pth (MB)         | 151.08       | 1109.38      | 94.24    | 288.54   |
| Checkpoint presente          | OK           | OK           | OK       | OK       |

### 2/4 — INFERÊNCIA (batch=4, crop=512×512, n=20)

| Métrica                      | Prithvi-Tiny | Prithvi-100M | UNet-R34 | UNet-R50 |
|------------------------------|--------------|--------------|----------|----------|
| ms / batch                   | 76.8         | 340.9        | 101.9    | 324.3    |
| ms / amostra                 | 19.2         | 85.2         | 25.5     | 81.1     |
| amostras / segundo           | 52.1         | 11.7         | 39.2     | 12.3     |
| pico GPU — inf (GB)          | 0.38         | 0.87         | 1.31     | 2.42     |
| speedup vs Prithvi-Tiny      | 1.00x        | 0.23x        | 0.75x    | 0.24x    |

### 3/4 — TREINO (métricas dos checkpoints)

| Métrica                      | Prithvi-Tiny | Prithvi-100M | UNet-R34 | UNet-R50 |
|------------------------------|--------------|--------------|----------|----------|
| Épocas treinadas             | 48           | 38           | 58       | 43       |
| Tempo médio / época (s)      | 8.1          | 32.3         | 13.0     | 208.5    |
| Tempo total treino (min)     | 6.5          | 20.5         | 12.6     | 149.3    |
| Mem GPU média treino (GB)    | 1.95         | 5.23         | 3.26     | 6.77     |
| Mem GPU pico treino (GB)     | 1.95         | 5.24         | 3.26     | 6.77     |
| LR final (média 3 ep.)       | 7.47e-06     | 1.93e-05     | —        | —        |
| Early stopping               | Não          | Não          | Sim (ep 58) | Não   |

### 4/4 — QUALIDADE (validação)

| Métrica                      | Prithvi-Tiny | Prithvi-100M | UNet-R34 | UNet-R50 |
|------------------------------|--------------|--------------|----------|----------|
| **Best mIoU**                | **0.6011**   | **0.6495**   | **0.2810** | **0.3005** |
| → época                      | 46           | 37           | 57       | 39       |
| Best pixel accuracy          | 0.8975       | 0.9089       | —        | —        |
| **Best val loss**            | **0.3851**   | **0.3457**   | **1.0235** | **1.0345** |
| → época                      | 48           | 38           | 48       | 43       |
| mIoU final (last epoch)      | 0.5935       | 0.6489       | 0.2805   | 0.2779   |
| Loss final (last epoch)      | 0.3851       | 0.3457       | 1.0339   | 1.0345   |

---

## Análise Comparativa

### Performance de Inferência
- **Prithvi-Tiny** é o mais rápido: 52.1 amostras/s, apenas 0.38 GB VRAM
- **UNet-R34** oferece bom compromisso: 39.2 amostras/s, 1.31 GB VRAM
- **Prithvi-100M** e **UNet-R50** são similares em velocidade (~12 amostras/s), mas R50 usa 2.8× mais VRAM

### Eficiência de Treino
- **Prithvi-Tiny**: mais rápido (8.1s/época), menor memória (1.95 GB)
- **UNet-R34**: 13s/época, 3.26 GB — extremamente eficiente
- **Prithvi-100M**: 32.3s/época, 5.23 GB — moderado
- **UNet-R50**: 208.5s/época, 6.77 GB — o mais pesado

### Qualidade Preditiva
- **Prithvi-100M**: melhor qualidade absoluta (mIoU 0.6495)
- **Prithvi-Tiny**: segundo lugar (mIoU 0.6011)
- **UNet-R50**: terceiro (mIoU 0.3005)
- **UNet-R34**: último (mIoU 0.2810)

### Trade-offs Chave
- **Prithvi foundation models** dominam em qualidade (≈2× melhor mIoU que ResNets) — diferença esperada quando se aproveita pré-treinamento HLS via `model_bands`
- **ResNets** são mais rápidos no treino, mas menos precisos no nosso domínio
- **Prithvi-Tiny** oferece o melhor custo-benefício geral
- **UNet-R50** não compensa o custo: treina 16× mais devagar que R34 para apenas 7% mais mIoU
