import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from data_loader import IHC_Benchmarking_Dataset
from models.pix2pix import UNetGenerator
from models.unet import UNetSegmenter

# --- 1. CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_A = r"D:\job\TESIS\data\processed_data\train_A"
PATH_B = r"D:\job\TESIS\data\processed_data\train_B"
M1_PATH = "../models/pix2pix_benchmarking.pth"
M2_PATH = "../models/unet_benchmarking.pth"


# --- 2. MÉTRICAS ---
def dice_coeff(pred, target, threshold=0.5):
    p = (pred > threshold).float()
    t = (target > threshold).float()
    inter = torch.sum(p * t)
    return (2.0 * inter) / (torch.sum(p) + torch.sum(t) + 1e-8)


# --- 3. CARGA DE MODELOS ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = IHC_Benchmarking_Dataset(PATH_A, PATH_B, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

m1 = UNetGenerator().to(DEVICE)
m1.load_state_dict(torch.load(M1_PATH))
m1.eval()

m2 = UNetSegmenter().to(DEVICE)
m2.load_state_dict(torch.load(M2_PATH))
m2.eval()

# --- 4. PROCESO DE EVALUACIÓN ---
metrics = {"M1": {"dice": [], "ssim": []}, "M2": {"dice": [], "ssim": []}}

print("Ejecutando Benchmarking Comparativo...")
with torch.no_grad():
    for i, (he, real_ihc, _) in enumerate(dataloader):
        he, real_ihc = he.to(DEVICE), real_ihc.to(DEVICE)

        # Inferencia
        out1 = m1(he)
        out2 = m2(he)

        # Cálculos de Métricas
        real_np = real_ihc[0].cpu().permute(1, 2, 0).numpy()
        p1_np = out1[0].cpu().permute(1, 2, 0).numpy()
        p2_np = out2[0].cpu().permute(1, 2, 0).numpy()

        metrics["M1"]["ssim"].append(ssim(real_np, p1_np, channel_axis=2, data_range=2.0))
        metrics["M2"]["ssim"].append(ssim(real_np, p2_np, channel_axis=2, data_range=2.0))
        metrics["M1"]["dice"].append(dice_coeff(out1, real_ihc).item())
        metrics["M2"]["dice"].append(dice_coeff(out2, real_ihc).item())

        # Visualización de la primera muestra
        if i == 0:
            plt.figure(figsize=(16, 4))
            display_list = [he[0], real_ihc[0], out1[0], out2[0]]
            titles = ['H&E Original', 'IHC Real (GT)', 'Pix2Pix (M1)', 'U-Net (M2)']
            for j in range(4):
                plt.subplot(1, 4, j + 1)
                plt.title(titles[j])
                plt.imshow(display_list[j].cpu().permute(1, 2, 0) * 0.5 + 0.5)
                plt.axis('off')
            plt.show()

# --- 5. REPORTE DE RESULTADOS ---
print("\n" + "=" * 40)
print("       REPORTE DE BENCHMARKING")
print("=" * 40)
for m_key, name in [("M1", "Pix2Pix"), ("M2", "U-Net")]:
    print(f"RESULTADOS {name}:")
    print(f"  > Promedio DICE: {np.mean(metrics[m_key]['dice']):.4f}")
    print(f"  > Promedio SSIM: {np.mean(metrics[m_key]['ssim']):.4f}")
    print("-" * 20)
print("=" * 40)