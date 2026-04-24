# Benchmark — Prithvi Tiny | Prithvi 100M | UNet R34 | UNet R50

**Dispositivo:** cuda  
**GPU:** NVIDIA GeForce RTX 3060 Laptop GPU  
**batch_size:** 4 | **n_batches inferência:** 20

---

## 1/4 — MODELO

| Métrica                      | Prithvi-Tiny | Prithvi-100M | UNet-R34 | UNet-R50 |
|------------------------------|--------------|--------------|----------|----------|
| Parâmetros totais (M)        | 13.2         | 96.9         | 24.7     | 75.5     |
| Parâmetros treináveis (M)    | 13.2         | 96.9         | 24.7     | 75.5     |
| Checkpoint .pth (MB)         | 151.08       | 1109.38      | 94.24    | 288.54   |
| Checkpoint presente          | ✓            | ✓            | ✓        | ✓        |

---

## 2/4 — INFERÊNCIA (batch=4, crop=512×512, n=20)

| Métrica                      | Prithvi-Tiny | Prithvi-100M | UNet-R34 | UNet-R50 |
|------------------------------|--------------|--------------|----------|----------|
| ms / batch                   | 76.8         | 340.9        | 101.9    | 324.3    |
| ms / amostra                 | 19.2         | 85.2         | 25.5     | 81.1     |
| amostras / segundo           | 52.1         | 11.7         | 39.2     | 12.3     |
| pico GPU — inf (GB)          | 0.38         | 0.87         | 1.31     | 2.42     |
| speedup vs Prithvi-Tiny      | 1.00×        | 0.23×        | 0.75×    | 0.24×    |

---

## 3/4 — TREINO (métricas dos checkpoints)

| Métrica                      | Prithvi-Tiny | Prithvi-100M | UNet-R34 | UNet-R50 |
|------------------------------|--------------|--------------|----------|----------|
| Épocas treinadas             | 48           | 38           | 58       | 43       |
| Tempo médio / época (s)      | 8.1          | 32.3         | 13.0     | 208.5    |
| Tempo total treino (min)     | 6.5          | 20.5         | 12.6     | 149.3    |
| Mem GPU média treino (GB)    | 1.95         | 5.23         | 3.26     | 6.77     |
| Mem GPU pico treino (GB)     | 1.95         | 5.24         | 3.26     | 6.77     |
| LR final (média 3 ep.)       | 7.47e-06     | 1.93e-05     | —        | —        |
| Early stopping               | Não          | Não          | Sim (ep 58) | Não   |

---

## 4/4 — QUALIDADE (validação)

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
- **🏆 Prithvi-100M**: melhor qualidade absoluta (mIoU 0.6495)
- **Prithvi-Tiny**: segundo lugar (mIoU 0.6011)
- **UNet-R50**: terceiro (mIoU 0.3005)
- **UNet-R34**: último (mIoU 0.2810)

### Trade-offs Chave
- **Prithvi foundation models** dominam em qualidade (2× melhor mIoU que ResNets)
- **ResNets** são mais rápidos no treino, mas menos precisos
- **Prithvi-Tiny** oferece o melhor custo-benefício geral
- **UNet-R50** não compensa o custo: treina 16× mais devagar que R34 para apenas 7% mais mIoU