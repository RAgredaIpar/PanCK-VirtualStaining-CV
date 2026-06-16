import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# Importaciones de tus arquitecturas
from models.unet import UNetSegmenter
from models.deepliif_net import DeepLIIFNet


class InferencePipeline:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.SRC_DIR = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = os.path.dirname(self.SRC_DIR)

        self.UNET_WEIGHTS = os.path.join(self.BASE_DIR, "models", "unet_benchmarking.pth")
        self.SEGMENTER_WEIGHTS = os.path.join(self.BASE_DIR, "models", "deepliif_segmenter.pth")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self._load_models()

    def _load_models(self):
        print(f"[+] Inicializando Pipeline Modular en: {self.device}")
        self.unet_pro = UNetSegmenter().to(self.device)
        self.unet_pro.load_state_dict(torch.load(self.UNET_WEIGHTS, map_location=self.device))
        self.unet_pro.eval()

        self.segmenter_sota = DeepLIIFNet().to(self.device)
        self.segmenter_sota.load_state_dict(torch.load(self.SEGMENTER_WEIGHTS, map_location=self.device))
        self.segmenter_sota.eval()
        print("[+] Modelos cargados y acoplados con éxito.")

    @torch.no_grad()
    def run_inference(self, img_path):
        """H&E -> IHC Sintética -> Máscara de Contornos"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No existe la imagen en: {img_path}")

        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        ihc_sintetica = self.unet_pro(img_tensor)
        mascara_contornos = self.segmenter_sota(ihc_sintetica)

        return img_tensor.squeeze(0), ihc_sintetica.squeeze(0), mascara_contornos.squeeze(0)

    def denormalize_to_numpy(self, tensor):
        """Convierte un tensor [-1, 1] de PyTorch a una imagen [0, 255] de OpenCV (RGB)"""
        array = tensor.cpu().numpy().transpose(1, 2, 0)
        array = ((array + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return array

    def analyze_cells(self, ihc_sintetica_tensor, mask_tensor):
        """
        MOTOR ANALÍTICO CALIBRADO (Fase 3): Añade un candado de saturación (S)
        para eliminar falsos positivos causados por tonos beige pálidos/lavados.
        """
        # 1. Convertir tensores a matrices numéricas de CPU
        img_ihc_rgb = self.denormalize_to_numpy(ihc_sintetica_tensor)
        img_mask_rgb = self.denormalize_to_numpy(mask_tensor)

        # De RGB de PyTorch a BGR de OpenCV, y luego a HSV
        img_ihc_bgr = cv2.cvtColor(img_ihc_rgb, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_ihc_bgr, cv2.COLOR_BGR2HSV)

        # Convertir la máscara a escala de grises y binarizar
        mask_gray = cv2.cvtColor(img_mask_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        # Encontrar contornos geométricos individuales
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pos_count = 0
        neg_count = 0
        audit_canvas = img_ihc_bgr.copy()

        # Clasificación Celular Objeto por Objeto
        for cnt in contours:
            if cv2.contourArea(cnt) < 15:
                continue

            cell_mask = np.zeros(mask_gray.shape, dtype=np.uint8)
            cv2.drawContours(cell_mask, [cnt], -1, 255, thickness=cv2.FILLED)

            # EXTRAER TONO (HUE) Y SATURACIÓN (SATURATION) de los píxeles internos
            hue_values = img_hsv[:, :, 0][cell_mask == 255]
            sat_values = img_hsv[:, :, 1][cell_mask == 255]  # <-- Canal 1 es Saturación en OpenCV

            if len(hue_values) == 0 or len(sat_values) == 0:
                continue

            mean_hue = np.mean(hue_values)
            mean_sat = np.mean(sat_values)  # <-- Promedio de intensidad del color en la célula

            # --- CLASIFICACIÓN CON CANDADO DE SATURACIÓN ---
            # El marrón DAB real de laboratorio es denso (Saturación alta).
            # Exigimos que mean_sat sea mayor o igual a 40 para evitar los grises/beiges lavados.
            if (10 <= mean_hue <= 32) and (mean_sat >= 32):
                pos_count += 1
                color_bgr = (0, 0, 255)  # Rojo para marcar células tumorales positivas (DAB+)
            else:
                neg_count += 1
                color_bgr = (255, 0, 0)  # Azul para marcar células sanas/estroma (Hematoxilina-)

            cv2.drawContours(audit_canvas, [cnt], -1, color_bgr, 1)

        # Cálculo del Índice Clínico Final
        total_cells = pos_count + neg_count
        positivity_index = (pos_count / total_cells * 100) if total_cells > 0 else 0.0

        results = {
            "total_nuclei": total_cells,
            "positive_nuclei": pos_count,
            "negative_nuclei": neg_count,
            "positivity_index": round(positivity_index, 2),
            "audit_image_rgb": cv2.cvtColor(audit_canvas, cv2.COLOR_BGR2RGB)
        }

        return results


if __name__ == "__main__":
    RUTA_TEST = r"D:\job\TESIS\data\processed_data\train_A\18_1.png"

    pipeline = InferencePipeline()
    try:
        he_t, ihc_t, mask_t = pipeline.run_inference(RUTA_TEST)

        # Ejecutar el nuevo Motor Analítico
        report = pipeline.analyze_cells(ihc_t, mask_t)

        print("\n" + "=" * 40)
        print("   REPORTE DEL MOTOR ANALÍTICO (PASO 3)")
        print("=" * 40)
        print(f"[+] Total de núcleos celulares detectados : {report['total_nuclei']}")
        print(f"[+] Células Inmunopositivas (Marrón/Tumor): {report['positive_nuclei']}")
        print(f"[+] Células Inmunonegativas (Azul/Sanas)  : {report['negative_nuclei']}")
        print(f"[+] ÍNDICE DE POSITIVIDAD CLINICO PAN-CK  : {report['positivity_index']}%")
        print("=" * 40)
        print("[+] Matriz de auditoría visual generada correctamente en memoria.")

    except Exception as e:
        print(f"[-] Error durante la ejecución del motor analítico: {e}")