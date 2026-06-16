import os
import torch
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
import wandb  # <-- Inyección de nuestra plataforma de auditoría

# Importamos tu pipeline del Paso 2 y 3
from pipeline_quantifier import InferencePipeline


def run_mass_validation_with_wandb(num_samples=50):
    TEST_DIR = r"D:\job\TESIS\data\raw\DeepLIIF_Testing_Set"

    if not os.path.exists(TEST_DIR):
        print(f"[-] ERROR: No se encontró la ruta: {TEST_DIR}")
        return

    # 1. INICIALIZAR SESIÓN DE AUDITORÍA EN WANDB
    print("[+] Conectando con Weights & Biases para auditoría visual...")
    run = wandb.init(
        project="Tesis-IHC-EsSalud",
        name="pipeline-validation-v1",
        job_type="model-evaluation"
    )

    # 2. CREAR LA ESTRUCTURA DE LA TABLA INTERACTIVA
    columns = [
        "ID_Parche",
        "Llamina_HE_Original",
        "IHC_Sintetica_UNetPRO",
        "Mascara_Contornos_G51",
        "Auditoria_Color_HSV",
        "Indice_Real_Lab",
        "Indice_Sintetico_IA",
        "Desviacion_Absoluta"
    ]
    wandb_table = wandb.Table(columns=columns)

    # Inicializar el backend de inferencia
    pipeline = InferencePipeline()

    all_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith('.png')])
    selected_files = all_files[:num_samples]

    real_indices = []
    fake_indices = []

    print("\n" + "-" * 60)
    print(f"{'Procesando Parche':<30} | {'Real':<8} | {'IA':<8}")
    print("-" * 60)

    # Crear un directorio temporal local para no ensuciar tu raíz
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_he_path = os.path.join(temp_dir, "temp_he.png")

    for filename in selected_files:
        img_path = os.path.join(TEST_DIR, filename)
        full_img = Image.open(img_path).convert("RGB")

        # Segmentación quirúrgica de la tira de DeepLIIF
        he_patch = full_img.crop((0, 0, 512, 512))
        ihc_real_patch = full_img.crop((1536, 0, 2048, 512))

        he_patch.save(temp_he_path)

        # --- INFERENCIA SINTÉTICA (TU IA) ---
        _, fake_ihc_tensor, fake_mask_tensor = pipeline.run_inference(temp_he_path)
        report_fake = pipeline.analyze_cells(fake_ihc_tensor, fake_mask_tensor)
        fake_index = report_fake["positivity_index"]

        # Convertir los lienzos de OpenCV de CPU a formato PIL para que WandB los procese
        audit_img_pil = Image.fromarray(report_fake["audit_image_rgb"])
        fake_ihc_pil = Image.fromarray(pipeline.denormalize_to_numpy(fake_ihc_tensor))
        mask_pil = Image.fromarray(pipeline.denormalize_to_numpy(fake_mask_tensor))

        # --- INFERENCIA REAL (CONTROL) ---
        ihc_real_tensor = pipeline.transform(ihc_real_patch).to(pipeline.device)
        with torch.no_grad():
            real_mask_tensor = pipeline.segmenter_sota(ihc_real_tensor.unsqueeze(0)).squeeze(0)
        report_real = pipeline.analyze_cells(ihc_real_tensor, real_mask_tensor)
        real_index = report_real["positivity_index"]

        # Guardar datos en vectores numéricos
        real_indices.append(real_index)
        fake_indices.append(fake_index)

        deviation = abs(real_index - fake_index)

        # 3. AGREGAR FILA CON IMÁGENES CENTRALIZADAS A LA TABLA DE WANDB
        wandb_table.add_data(
            filename,
            wandb.Image(he_patch),
            wandb.Image(fake_ihc_pil),
            wandb.Image(mask_pil),
            wandb.Image(audit_img_pil),
            real_index,
            fake_index,
            round(deviation, 2)
        )

        short_name = filename if len(filename) <= 30 else filename[:27] + "..."
        print(f"{short_name:<30} | {real_index:<6}% | {fake_index:<6}%")

    # Limpieza de residuo temporal
    if os.path.exists(temp_he_path):
        os.remove(temp_he_path)
    os.rmdir(temp_dir)

    # --- 4. TRATAMIENTO ESTADÍSTICO FINAL ---
    real_indices = np.array(real_indices)
    fake_indices = np.array(fake_indices)

    r_coef, p_value = pearsonr(real_indices, fake_indices)
    mae = mean_absolute_error(real_indices, fake_indices)
    r2 = r2_score(real_indices, fake_indices)

    # 5. REGISTRAR MÉTRICAS FINALES Y TABLA EN LA NUBE
    wandb.log({
        "Métricas_Validación/Pearson_r": r_coef,
        "Métricas_Validación/R2_Score": r2,
        "Métricas_Validación/MAE_Porcentual": mae,
        "Métricas_Validación/p_value": p_value,
        "Tablas_Auditoria/Lote_Control_50_Parches": wandb_table
    })

    print("\n" + "=" * 50)
    print("   MÉTRICAS CARGADAS EXITOSAMENTE A WANDB")
    print("=" * 50)
    print(f"[+] Pearson r : {r_coef:.4f}")
    print(f"[+] MAE       : {mae:.2f}%")
    print("=" * 50)

    run.finish()


if __name__ == "__main__":
    run_mass_validation_with_wandb(num_samples=50)