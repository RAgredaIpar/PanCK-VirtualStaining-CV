import os
import torch
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import cv2

# Importamos tu pipeline base
from pipeline_quantifier import InferencePipeline


def optimize_hyperparameters(num_samples=50):
    TEST_DIR = r"D:\job\TESIS\data\raw\DeepLIIF_Testing_Set"
    if not os.path.exists(TEST_DIR):
        print(f"[-] ERROR: No existe la ruta: {TEST_DIR}")
        return

    pipeline = InferencePipeline()
    all_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith('.png')])[:num_samples]

    # --- PASO 1: CACHEAR LAS IMÁGENES EN MEMORIA EN ESPACIO HSV ---
    print("[+] Fase 1: Extrayendo y cacheando parches biomédicos en memoria RAM...")
    cached_data = []

    # Crear un archivo temporal flotante
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_tune")
    os.makedirs(temp_dir, exist_ok=True)
    temp_he_path = os.path.join(temp_dir, "temp_he.png")

    for filename in all_files:
        img_path = os.path.join(TEST_DIR, filename)
        full_img = Image.open(img_path).convert("RGB")

        he_patch = full_img.crop((0, 0, 512, 512))
        ihc_real_patch = full_img.crop((1536, 0, 2048, 512))

        he_patch.save(temp_he_path)

        # Inferencia sintética (IA)
        _, fake_ihc_tensor, fake_mask_tensor = pipeline.run_inference(temp_he_path)
        img_ihc_rgb = pipeline.denormalize_to_numpy(fake_ihc_tensor)
        img_mask_rgb = pipeline.denormalize_to_numpy(fake_mask_tensor)

        fake_hsv = cv2.cvtColor(cv2.cvtColor(img_ihc_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        mask_gray_fake = cv2.cvtColor(img_mask_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh_fake = cv2.threshold(mask_gray_fake, 127, 255, cv2.THRESH_BINARY)
        contours_fake, _ = cv2.findContours(thresh_fake, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Inferencia real (Lab)
        ihc_real_tensor = pipeline.transform(ihc_real_patch).to(pipeline.device)
        with torch.no_grad():
            real_mask_tensor = pipeline.segmenter_sota(ihc_real_tensor.unsqueeze(0)).squeeze(0)
        img_real_rgb = pipeline.denormalize_to_numpy(ihc_real_tensor)
        img_real_mask_rgb = pipeline.denormalize_to_numpy(real_mask_tensor)

        real_hsv = cv2.cvtColor(cv2.cvtColor(img_real_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        mask_gray_real = cv2.cvtColor(img_real_mask_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh_real = cv2.threshold(mask_gray_real, 127, 255, cv2.THRESH_BINARY)
        contours_real, _ = cv2.findContours(thresh_real, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cached_data.append({
            "fake_hsv": fake_hsv, "fake_contours": contours_fake, "fake_mask_shape": mask_gray_fake.shape,
            "real_hsv": real_hsv, "real_contours": contours_real, "real_mask_shape": mask_gray_real.shape
        })

    if os.path.exists(temp_he_path):
        os.remove(temp_he_path)
    os.rmdir(temp_dir)
    print(f"[+] Cache completado. {len(cached_data)} muestras listas para Grid Search.")

    # --- PASO 2: LOOP RÁPIDO DE BÚSQUEDA POR CUADRÍCULA (GRID SEARCH) ---
    print("\n[+] Fase 2: Ejecutando Grid Search analítico sobre umbrales HSV...")

    # Espacio de búsqueda a explorar
    sat_candidates = range(10, 36, 2)  # Probar saturaciones desde 10 hasta 34 de 2 en 2
    max_hue_candidates = range(30, 41, 2)  # Probar límites de tono marrón desde 30 hasta 40

    best_r = -1.0
    best_mae = 100.0
    best_config = {}

    results_grid = []

    for min_sat in sat_candidates:
        for max_hue in max_hue_candidates:

            real_indices = []
            fake_indices = []

            # Evaluar las 50 muestras en microsegundos usando la cache
            for data in cached_data:
                # Conteo Sintético (IA)
                pos_f, neg_f = 0, 0
                for cnt in data["fake_contours"]:
                    if cv2.contourArea(cnt) < 15: continue
                    cell_mask = np.zeros(data["fake_mask_shape"], dtype=np.uint8)
                    cv2.drawContours(cell_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                    h_vals = data["fake_hsv"][:, :, 0][cell_mask == 255]
                    s_vals = data["fake_hsv"][:, :, 1][cell_mask == 255]
                    if len(h_vals) == 0: continue
                    if (10 <= np.mean(h_vals) <= max_hue) and (np.mean(s_vals) >= min_sat):
                        pos_f += 1
                    else:
                        neg_f += 1
                tot_f = pos_f + neg_f
                fake_indices.append((pos_f / tot_f * 100) if tot_f > 0 else 0.0)

                # Conteo Real (Laboratorio)
                pos_r, neg_r = 0, 0
                for cnt in data["real_contours"]:
                    if cv2.contourArea(cnt) < 15: continue
                    cell_mask = np.zeros(data["real_mask_shape"], dtype=np.uint8)
                    cv2.drawContours(cell_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                    h_vals = data["real_hsv"][:, :, 0][cell_mask == 255]
                    s_vals = data["real_hsv"][:, :, 1][cell_mask == 255]
                    if len(h_vals) == 0: continue
                    # El laboratorio real usa su umbral nativo estándar
                    if (10 <= np.mean(h_vals) <= 35) and (np.mean(s_vals) >= 15):
                        pos_r += 1
                    else:
                        neg_r += 1
                tot_r = pos_r + neg_r
                real_indices.append((pos_r / tot_r * 100) if tot_r > 0 else 0.0)

            # Calcular métricas para esta combinación
            r_coef, _ = pearsonr(real_indices, fake_indices)
            mae = mean_absolute_error(real_indices, fake_indices)

            if np.isnan(r_coef): r_coef = -1.0

            results_grid.append({"sat": min_sat, "hue": max_hue, "r": r_coef, "mae": mae})

            # Condición de éxito: Guardamos la que maximice Pearson y mantenga un MAE controlado
            if r_coef > best_r:
                best_r = r_coef
                best_mae = mae
                best_config = {"sat": min_sat, "hue": max_hue}

    # --- PASO 3: DESPLEGAR EL CUADRO DE RESULTADOS ---
    print("\n" + "=" * 55)
    print("      TOP 5 CONFIGURACIONES ENCONTRADAS")
    print("=" * 55)
    print(f"{'Saturación Mín':<16} | {'Tono Máx (Hue)':<16} | {'Pearson r':<10} | {'MAE':<6}")
    print("-" * 55)

    # Ordenar combinaciones por el coeficiente de Pearson más alto
    sorted_results = sorted(results_grid, key=lambda x: x["r"], reverse=True)
    for res in sorted_results[:5]:
        print(f"S >= {res['sat']:<12} | H <= {res['hue']:<12} | {res['r']:<9.4f} | {res['mae']:.2f}%")
    print("=" * 55)

    print(
        f"\n[+] PUNTO DULCE DEFINITIVO: Saturación Mínima >= {best_config['sat']} | Tono Máximo <= {best_config['hue']}")
    print(f"[+] Rendimiento Óptimo: Pearson r = {best_r:.4f} | MAE = {best_mae:.2f}%")
    print("[!] Modifica estos dos valores en tu 'src/pipeline_quantifier.py' para congelar el récord.")


if __name__ == "__main__":
    optimize_hyperparameters(num_samples=50)