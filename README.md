# Satellite Segmentation Framework

Este projeto é um framework para segmentação semântica de imagens de satélite, integrando modelos de última geração como o **Prithvi** (Geospatial Foundation Model da IBM/NASA) e a arquitetura clássica **U-Net**. O foco é facilitar o treinamento, avaliação e exportação de modelos para produção.

## 📂 Estrutura do Repositório

### 1. `scripts/` (Entry Points)
* **`prithvi/`**: Implementações específicas para o modelo Prithvi.
    * `train_prithvi.py`: Orquestra o treinamento utilizando pesos pré-treinados do ViT.
    * `prithvi_fp16.py`: Versão otimizada para inferência em precisão mista.
    * `predict_prithvi`: Inferencia usando prithvi, edite as constantes no codigo
* **`unet/`**: Workflow para a arquitetura U-Net.
    * `export_onnx_unet.py`: Script para converter o modelo `.pt` para **ONNX**, ideal para deploy em ambientes C++ ou Web.
    * `train_unet`
    * `predict_unet`
---

## ⚙️ Configuração (`config/`)

O projeto utiliza arquivos **YAML** para centralizar todos os hiperparâmetros. Isso permite reprodutibilidade sem alterar o código-fonte.


## 🛠️ Configuração no Windows

Trabalhar com dados geoespaciais no Windows exige atenção à biblioteca `PROJ`, que o `rasterio` utiliza para sistemas de coordenadas.

### 1. Ambiente Virtual
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Variável de Ambiente Crítica
Sempre execute este comando antes de iniciar o treino para evitar erros de `PROJ_DATA`:
```powershell
$env:PROJ_LIB = "D:\Projects\SatSegmentation\venv\Lib\site-packages\rasterio\proj_data"
```

---

## 🚀 Execução

### Treinamento
Para iniciar o treinamento do modelo Prithvi, utilize o módulo correspondente. O uso do flag `-m` garante que o Python resolva corretamente os imports da pasta `src`.

```powershell
python -m scripts.prithvi.train_prithvi --config config/prithvi.yaml
```
```powershell
python -m scripts.unet.train_unet --config config/unet.yaml
```

### Inferência e Predição
```powershell
### Primeiro rode prithvi_fp16 para ter o modelo otimizado e mude o path no codigo
python -m scripts.prithvi.predict_prithvi

### Exportação para Produção (ONNX)
Para converter o modelo U-Net para alta performance:
```powershell
python -m scripts.unet.export_onnx_unet 
```

---

## 📊 Métricas e Validação
Os logs de treinamento incluem:
* **mIoU (mean Intersection over Union)**: Principal métrica de performance de segmentação.
* **Loss Curve**: Monitoramento de convergência.
* **Visualização de Máscaras**: Exportação periódica de predições vs Ground Truth para `eval.py`.

