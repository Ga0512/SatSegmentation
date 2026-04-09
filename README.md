# 🛰️ Satellite Segmentation Framework

Framework simplificado para segmentação de imagens de satélite usando **Prithvi (IBM/NASA)** e **U-Net**.

## 🚀 Setup Rápido (Windows)

```powershell
https://github.com/Ga0512/SatSegmentation.git
cd SatSegmentation
```

1. **Dependências:** 

```powershell
python -m venv venv
venv/scripts/activate
python.exe -m pip install --upgrade pip
python -m src.fix_terratorch   
```


Instale o [GDAL wheel](https://wheelhouse.openquake.org/v3/windows/py310/) e o PyTorch:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

2. **Variável de Ambiente:** Execute sempre antes de iniciar:
```powershell
$env:PROJ_LIB = "$pwd\venv\Lib\site-packages\rasterio\proj_data"
```

---

## 🛠️ Execução

O framework utiliza arquivos `.yaml` em `config/` para todos os parâmetros.

### 1. Treinamento
```powershell
# Prithvi (Geospatial Foundation Model)
python -m scripts.prithvi.train_prithvi --config config/prithvi.yaml

# U-Net (Clássica)
python -m scripts.unet.train_unet --config config/unet.yaml
```

### 2. Inferência e Otimização
* **Prithvi FP16:** Use o script de precisão mista para inferência ultra rápida:
  ```powershell
  python -m scripts.prithvi.prithvi_fp16
  ```
* **Predição Simples:** `python -m scripts.prithvi.predict_prithvi`
* **Exportação ONNX:** Converta a U-Net para produção:
  ```powershell
  python -m scripts.unet.export_onnx_unet
  ```

---

## 📂 Estrutura Minimalista

* **`config/`**: Hiperparâmetros (YAML).
* **`scripts/`**: Entry points (Treino, Predição e Exportação).
* **`src/`**: Core do framework (Modelos e DataLoaders).

## 📊 Métricas
O progresso é validado via **mIoU**, curvas de **Loss** e exportação de máscaras comparativas para inspeção visual.
