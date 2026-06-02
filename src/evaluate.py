import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import os

from data_loader import IHC_Benchmarking_Dataset
from models.pix2pix import UNetGenerator
from models.unet import UNetSegmenter
from models.deepliif_net import DeepLIIFResNetGenerator

# --- 1. CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_A = r"D:\job\TESIS\data\processed_data\train_A"
PATH_B = r"D:\job\TESIS\data\processed_data\train_B"

M1_PATH = "../models/pix2pix_benchmarking.pth"
M2_PATH = "../models/unet_benchmarking.pth"
M3_PATH = "../models/deepliif_official.pth"


def dice_coeff(pred, target, threshold=0.5):
    p = (pred > threshold).float()
    t = (target > threshold).float()
    inter = torch.sum(p * t)
    return (2.0 * inter) / (torch.sum(p) + torch.sum(t) + 1e-8)


# --- 2. DATA PIPELINE ---
dataset = IHC_Benchmarking_Dataset(PATH_A, PATH_B, is_train=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# --- 3. CARGA DE MODELOS ---
print("Cargando contendientes...")
m1 = UNetGenerator().to(DEVICE)
m1.load_state_dict(torch.load(M1_PATH, map_location=DEVICE))
m1.eval()

m2 = UNetSegmenter().to(DEVICE)
m2.load_state_dict(torch.load(M2_PATH, map_location=DEVICE))
m2.eval()

m3_available = False
m3 = DeepLIIFResNetGenerator().to(DEVICE)
if os.path.exists(M3_PATH):
    try:
        m3.load_state_dict(torch.load(M3_PATH, map_location=DEVICE))
        m3.eval()
        m3_available = True
        print("Modelo SOTA (DeepLIIF) cargado con éxito.")
    except Exception as e:
        print(f"Error al cargar pesos de DeepLIIF: {e}")
else:
    print(f"Aviso: No se encontró {M3_PATH}. Evaluando solo M1 y M2.")

# --- 4. EVALUACIÓN Y ACUMULADORES CLÍNICOS ---
models_to_eval = [("M1", "Pix2Pix"), ("M2", "U-Net PRO")]
if m3_available:
    models_to_eval.append(("M3", "DeepLIIF SOTA"))

metrics = {m[0]: {"dice": [], "ssim": []} for m in models_to_eval}

# Acumuladores de píxeles para la Matriz de Confusión de tu modelo principal (M2)
total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

print("\nEjecutando Benchmarking...")
with torch.no_grad():
    for i, (he, real_ihc, _) in enumerate(dataloader):
        he, real_ihc = he.to(DEVICE), real_ihc.to(DEVICE)

        # Inferencia
        outs = {"M1": m1(he), "M2": m2(he)}
        if m3_available:
            outs["M3"] = m3(he)

        # Desnormalizar imágenes a rango estándar [0, 1] en formato NumPy HWC
        real_np = real_ihc[0].cpu().permute(1, 2, 0).numpy()
        real_rgb = real_np * 0.5 + 0.5

        # --- EXTRACCIÓN DE MATRIZ DE CONFUSIÓN CLÍNICA (PÍXEL-LEVEL para M2) ---
        pred_m2_rgb = outs["M2"][0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5

        # Filtro delta cromático: Si el canal Rojo supera al Azul por un margen, es Marrón (DAB+)
        mask_real = (real_rgb[:, :, 0] - real_rgb[:, :, 2]) > 0.12
        mask_pred = (pred_m2_rgb[:, :, 0] - pred_m2_rgb[:, :, 2]) > 0.12

        # Comparación de máscaras de positividad
        total_tp += np.sum((mask_real == 1) & (mask_pred == 1))
        total_tn += np.sum((mask_real == 0) & (mask_pred == 0))
        total_fp += np.sum((mask_real == 0) & (mask_pred == 1))
        total_fn += np.sum((mask_real == 1) & (mask_pred == 0))

        # Cálculos de Métricas Estándar
        for m_key, _ in models_to_eval:
            pred_tensor = outs[m_key]
            pred_np = pred_tensor[0].cpu().permute(1, 2, 0).numpy()

            metrics[m_key]["ssim"].append(ssim(real_np, pred_np, channel_axis=2, data_range=2.0))
            metrics[m_key]["dice"].append(dice_coeff(pred_tensor, real_ihc).item())

        # Visualización de la primera muestra
        if i == 0:
            num_cols = 4 if not m3_available else 5
            plt.figure(figsize=(4 * num_cols, 4))

            display_list = [he[0], real_ihc[0], outs["M1"][0], outs["M2"][0]]
            titles = ['H&E Original', 'IHC Real', 'Pix2Pix (M1)', 'U-Net PRO (M2)']

            if m3_available:
                display_list.append(outs["M3"][0])
                titles.append('DeepLIIF SOTA (M3)')

            for j in range(num_cols):
                plt.subplot(1, num_cols, j + 1)
                plt.title(titles[j])
                plt.imshow(display_list[j].cpu().permute(1, 2, 0) * 0.5 + 0.5)
                plt.axis('off')
            plt.show()

        # Nota: Puedes aumentar el número a 100 o quitar el break si deseas evaluar el subset completo
        if i >= 50:
            break

# --- 5. REPORTE FINAL ---
print("\n" + "=" * 45)
print("        REPORTE DE BENCHMARKING FINAL")
print("=" * 45)
for m_key, name in models_to_eval:
    print(f"RESULTADOS {name}:")
    print(f"  > Promedio DICE: {np.mean(metrics[m_key]['dice']):.4f}")
    print(f"  > Promedio SSIM: {np.mean(metrics[m_key]['ssim']):.4f}")
    print("-" * 25)

# Cálculo formal de las tasas probabilísticas de idoneidad clínica para M2
sensibilidad = total_tp / (total_tp + total_fn + 1e-8)
especificidad = total_tn / (total_tn + total_fp + 1e-8)
vpp = total_tp / (total_tp + total_fp + 1e-8)

print("MÉTRICAS DE CONCORDANCIA DIAGNÓSTICA REAL (U-Net PRO):")
print(f"  > Conteo Bruto Píxeles -> TP: {total_tp} | TN: {total_tn} | FP: {total_fp} | FN: {total_fn}")
print(f"  > Sensibilidad Clínica : {sensibilidad * 100:.2f}%")
print(f"  > Especificidad Clínica: {especificidad * 100:.2f}%")
print(f"  > Valor Predictivo Pos.: {vpp * 100:.2f}%")
print("=" * 45)