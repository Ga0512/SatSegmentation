import torch
import logging
from src.model import PrithviSegmentation

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def export_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './output/best_model_prithvi.pth'
    onnx_path = './model/prithvi.onnx'
    num_bands = 11
    num_classes = 19
    tile_size = 512

    logging.info('Carregando modelo PyTorch...')
    model = PrithviSegmentation(in_channels=num_bands, num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    # Tensor de exemplo (Batch=1, mas será dinâmico)
    dummy_input = torch.randn(1, num_bands, tile_size, tile_size, device=device)

    logging.info('Exportando para ONNX...')
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14, # Opset 14 é estável e tem bom suporte
        do_constant_folding=True, # Otimiza constantes no grafo
        input_names=['input'], 
        output_names=['logits'],
        # Eixos dinâmicos permitem mudar o batch_size na hora da inferência
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'logits': {0: 'batch_size'}
        }
    )
    logging.info(f'Modelo exportado com sucesso para: {onnx_path}')

if __name__ == '__main__':
    export_to_onnx()
    from onnxconverter_common import float16
    import onnx

    model = onnx.load("./model/prithvi.onnx")
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, "./model/prithvi_fp16.onnx")