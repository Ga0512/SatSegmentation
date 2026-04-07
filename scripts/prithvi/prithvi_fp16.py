import os
import torch
from src.model import Prithvi11BandsModel

# --- CONFIGURAÇÕES ---
os.environ["PROJ_DATA"] = r"..\venv\Lib\site-packages\rasterio\proj_data"
MODEL_PATH = './model/best_prithvi_11bands.pth'
OUTPUT_FP16_PATH = './model/best_prithvi_11bands_fp16.pth'
OUTPUT_STANDALONE_PATH = './model/prithvi_production_fp16.pt' # TorchScript
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 19
NUM_BANDS = 11

def save_right_and_complete():
    print("1. Inicializando modelo em FP32...")
    model = Prithvi11BandsModel(NUM_CLASSES, NUM_BANDS, pretrained=False)
    
    # Carrega os pesos originais
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.to(DEVICE)
    model.eval()

    print("2. Convertendo para FP16 (Half Precision)...")
    # Reduz o uso de VRAM pela metade e acelera inferência em GPUs com Tensor Cores
    model.half()

    print(f"3. Salvando State Dict em FP16: {OUTPUT_FP16_PATH}")
    # Salva apenas os pesos. Mais seguro e compatível.
    torch.save(model.state_dict(), OUTPUT_FP16_PATH)

    print(f"4. Criando arquivo Standalone (TorchScript)...")
    # Isso empacota o grafo + pesos. Útil para deploy sem precisar da classe src.model
    try:
        # Usamos trace porque o Prithvi tem shapes de entrada bem definidos (ex: 512x512)
        dummy_input = torch.randn(1, NUM_BANDS, 1, 512, 512, dtype=torch.float16, device=DEVICE)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(OUTPUT_STANDALONE_PATH)
        print(f"✅ Sucesso! Modelo portável salvo em: {OUTPUT_STANDALONE_PATH}")
    except Exception as e:
        print(f"⚠️ Erro ao criar TorchScript (comum em arquiteturas complexas): {e}")

    print("\n🚀 PROCESSO CONCLUÍDO!")

if __name__ == "__main__":
    save_right_and_complete()