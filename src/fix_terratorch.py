import os
import sys

def apply_terratorch_patch():
    # Detecta o caminho do site-packages no seu venv ativo
    # Isso evita problemas de caminhos absolutos (C: vs D:)
    site_packages = [
        p for p in sys.path 
        if 'site-packages' in p or 'dist-packages' in p
    ]
    
    if not site_packages:
        print("❌ Erro: Não consegui encontrar a pasta 'site-packages' do venv.")
        print("Certifique-se de que o venv está ATIVO antes de rodar este script.")
        return

    target_file = os.path.join(
        site_packages[0], 
        'terratorch', 'models', 'backbones', 'torchgeo_resnet.py'
    )

    if not os.path.exists(target_file):
        print(f"❌ Erro: Arquivo não encontrado em:\n{target_file}")
        return

    print(f"🔍 Localizado: {target_file}")

    # Lendo o conteúdo
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Verificando se a correção é necessária
    old_text = "SENTINEL2_ALL_SOFTCON"
    new_text = "SENTINEL2_ALL_MOCO"

    if old_text in content:
        new_content = content.replace(old_text, new_text)
        
        # Salvando a alteração
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Sucesso! '{old_text}' foi substituído por '{new_text}'.")
    elif new_text in content:
        print("ℹ️ O arquivo já parece estar corrigido (MOCO detectado).")
    else:
        print("⚠️ Aviso: Não encontrei a string alvo. Talvez a versão seja diferente?")

if __name__ == "__main__":
    apply_terratorch_patch()