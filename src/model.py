import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import terratorch  
from terratorch.models import EncoderDecoderFactory


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# AttentionResUNet
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, 1, bias=False)

        self.spatial = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(F_int, 1, 1, bias=False), nn.Sigmoid())
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(F_l, F_l//16, 1), nn.ReLU(inplace=True), nn.Conv2d(F_l//16, F_l, 1), nn.Sigmoid())

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.spatial(g1 + x1)

        x_spatial = x * psi
        ch = self.channel(x_spatial)
        x_channel = x_spatial * ch

        return x_channel + x


# (init_ch, enc1, enc2, enc3, enc4)
_ENCODER_CH = {
    'resnet34': (64,  64,  128,  256,   512),
    'resnet50': (64, 256,  512, 1024,  2048),
}


class AttentionResUNet(nn.Module):
    def __init__(self, in_channels, num_classes, model_size='resnet34'):
        super().__init__()
        assert model_size in _ENCODER_CH, f"model_size deve ser um de: {list(_ENCODER_CH)}"
        ci, e1, e2, e3, e4 = _ENCODER_CH[model_size]

        if model_size == 'resnet34':
            enc = models.resnet34(weights=None)
        else:
            enc = models.resnet50(weights=None)

        old = enc.conv1
        enc.conv1 = nn.Conv2d(in_channels, old.out_channels, 7, stride=2, padding=3, bias=False)

        self.initial = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool    = enc.maxpool
        self.enc1    = enc.layer1
        self.enc2    = enc.layer2
        self.enc3    = enc.layer3
        self.enc4    = enc.layer4

        self.up4  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att4 = AttentionGate(e4, e3, e3 // 2)
        self.dec4 = self.block(e4 + e3, e3)

        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att3 = AttentionGate(e3, e2, e2 // 2)
        self.dec3 = self.block(e3 + e2, e2)

        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att2 = AttentionGate(e2, e1, e1 // 2)
        self.dec2 = self.block(e2 + e1, e1)

        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = self.block(e1 + ci, 64)

        self.final = nn.Conv2d(64, num_classes, 1)


    def block(self,i,o):
        return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


    def forward(self,x):
        x0 = self.initial(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        u4 = self.up4(x4)
        a4 = self.att4(u4,x3)
        d4 = self.dec4(torch.cat([u4,a4],1))

        u3 = self.up3(d4)
        a3 = self.att3(u3,x2)
        d3 = self.dec3(torch.cat([u3,a3],1))

        u2 = self.up2(d3)
        a2 = self.att2(u2,x1)
        d2 = self.dec2(torch.cat([u2,a2],1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1,x0],1))

        out = self.final(d1)

        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out


# model_size → (backbone_name, select_indices, decoder_channels)
_PRITHVI_SIZES = {
    'tiny': ('prithvi_eo_v2_tiny_tl', [2, 5, 8, 11],  256),
    '100m': ('prithvi_eo_v2_100_tl',  [2, 5, 8, 11],  256),
    '300m': ('prithvi_eo_v2_300_tl',  [5, 11, 17, 23], 256),
    '600m': ('prithvi_eo_v2_600_tl',  [7, 15, 23, 31], 256),
}


class Prithvi11BandsModel(nn.Module):
    '''
    model_size  backbone               blocos  embed
    tiny        prithvi_eo_v2_tiny_tl  12      192
    100m        prithvi_eo_v2_100_tl   12      768
    300m        prithvi_eo_v2_300_tl   24      1024
    600m        prithvi_eo_v2_600_tl   32      1280
    '''
    def __init__(self, num_classes, num_bands=11, pretrained=True, model_size='tiny'):
        super().__init__()
        assert model_size in _PRITHVI_SIZES, f"model_size deve ser um de: {list(_PRITHVI_SIZES)}"
        backbone_name, select_indices, decoder_channels = _PRITHVI_SIZES[model_size]
        factory = EncoderDecoderFactory()
        self.base = factory.build_model(
            task="segmentation",
            backbone=backbone_name,
            backbone_pretrained=pretrained,
            backbone_in_chans=6,
            backbone_num_frames=1,
            decoder="UperNetDecoder",
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            necks=[{"name": "SelectIndices", "indices": select_indices}, {"name": "ReshapeTokensToImage"}]
        )
        
        old_proj = self.base.encoder.patch_embed.proj
        
        # Cria a nova convolução 3D com o novo número de bandas
        self.new_proj = nn.Conv3d(
            in_channels=num_bands, 
            out_channels=old_proj.out_channels, 
            kernel_size=old_proj.kernel_size, 
            stride=old_proj.stride,
            padding=old_proj.padding, # Boa prática incluir padding se houver
            bias=(old_proj.bias is not None)
        )
        
        if pretrained:
            with torch.no_grad():
                # 1. Copia os 6 canais originais (HLS)
                n_original_bands = old_proj.in_channels # que é 6
                self.new_proj.weight[:, :n_original_bands, :, :, :] = old_proj.weight.clone()
                
                # 2. Calcula a média dos pesos originais ao longo da dimensão dos canais
                mean_weight = old_proj.weight.mean(dim=1, keepdim=True)
                
                # 3. Preenche as bandas extras (da 7 até a 11) com a média expandida
                extras = num_bands - n_original_bands
                if extras > 0:
                    self.new_proj.weight[:, n_original_bands:, :, :, :] = mean_weight.expand(-1, extras, -1, -1, -1)
                
                # 4. Copia o bias (Viés) se existir
                if old_proj.bias is not None:
                    self.new_proj.bias.copy_(old_proj.bias)
                
        # Substitui a camada no modelo base
        self.base.encoder.patch_embed.proj = self.new_proj

    def forward(self, x):
        # O Prithvi espera entrada 5D: (Batch, Channels, Time, Height, Width)
        if x.dim() == 4: 
            x = x.unsqueeze(2) # Adiciona a dimensão temporal (T=1)
            
        # Dependendo da versão do TerraTorch, o retorno pode ser um dicionário ou o tensor direto.
        res = self.base(x)
        
        # Se 'res' for um objeto customizado ou dicionário com atributo 'output'
        out = res.output if hasattr(res, 'output') else res
        
        # Garante que o output tenha o mesmo tamanho espacial da entrada (H, W)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        return out
