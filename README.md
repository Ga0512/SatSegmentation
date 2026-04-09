# 🛰️ Satellite Segmentation Framework

Framework simplificado para segmentação de imagens de satélite usando **Prithvi (IBM/NASA)** e **U-Net**.

## 🚀 Setup Rápido Windows

```powershell
git clone https://github.com/Ga0512/SatSegmentation.git
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

Aqui está o seu trecho formatado com uma estrutura mais limpa, profissional e fácil de seguir. Organizei os comandos para que o fluxo de instalação e execução faça sentido imediato para quem estiver lendo o seu `README.md`.

---

## 🚀 Setup Rápido: Docker Container

Siga os passos abaixo para preparar o ambiente e colocar o container para rodar com suporte a GPU.

### 🐳 1. Pré-requisitos
Antes de começar, certifique-se de ter instalado:
* **Docker**
* **NVIDIA Container Toolkit** (Essencial para habilitar o uso da GPU dentro do Docker)

---

### 🛠️ 2. Instalação (Windows + WSL ou Linux)
Execute os comandos abaixo para instalar o toolkit da NVIDIA e reiniciar o serviço do Docker:

```bash
# Atualiza os repositórios e instala o toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Reinicia o Docker para aplicar as mudanças
sudo systemctl restart docker
```

**Teste a instalação:**
Verifique se o Docker consegue acessar sua GPU com o comando:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

---

### 📦 3. Build e Execução
Na raiz do projeto, execute os comandos para construir a imagem e subir o container:

**Build da imagem:**
```bash
docker build -t satseg .
```

**Rodar o container:**
> **Nota:** O comando abaixo mapeia sua pasta atual para dentro do container e habilita todas as GPUs.
```bash
docker run -it --gpus all -v ${PWD}:/app satseg
```

---

### 🧪 4. Treinamento e Ajustes
Dentro do container, primeiro aplique o fix necessário e depois escolha o modelo para treinar:

**Ajuste inicial:**
```bash
python -m src.fix_terratorch
```

**Comandos de Treinamento:**
```powershell
# Prithvi (Geospatial Foundation Model)
python -m scripts.prithvi.train_prithvi --config config/prithvi.yaml

# U-Net (Clássica)
python -m scripts.unet.train_unet --config config/unet.yaml
```

---

### 💡 Dica rápida
Se estiver usando **Windows (PowerShell)**, o comando `${PWD}` funciona perfeitamente para montar o volume. Se estiver no **CMD**, utilize `%cd%`. No **Linux**, use `$(pwd)`.

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
