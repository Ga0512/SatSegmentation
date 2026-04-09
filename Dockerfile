FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# evitar prompts
ENV DEBIAN_FRONTEND=noninteractive

# dependências básicas
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# define python padrão
RUN ln -s /usr/bin/python3 /usr/bin/python

# working dir
WORKDIR /app

# copia projeto
COPY . .

# pip upgrade
RUN pip install --upgrade pip

# torch com CUDA 12.1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# requirements
RUN pip install -r requirements.txt

# fix terratorch
RUN python -m src.fix_terratorch

# variável do rasterio
ENV PROJ_LIB=/usr/local/lib/python3.10/dist-packages/rasterio/proj_data

# default command
CMD ["/bin/bash"]