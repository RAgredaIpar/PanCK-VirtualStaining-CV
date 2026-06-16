import os
import wandb

def upload_segmenter_artifact():
    # --- 1. CONFIGURACIÓN DE RUTAS ---
    # Calculamos la ruta absoluta de la raíz del proyecto para evitar fallos de ejecución en Windows
    SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR = os.path.dirname(SRC_DIR)
    MODEL_PATH = os.path.join(BASE_DIR, "models", "deepliif_segmenter.pth")

    if not os.path.exists(MODEL_PATH):
        print(f"[-] ERROR: No se encontró el archivo en: {MODEL_PATH}")
        print("[!] Asegúrate de haber copiado 'latest_net_G51.pth' a 'models/' y renombrarlo como 'deepliif_segmenter.pth'.")
        return

    # --- 2. INICIALIZAR ENTORNO WANDB ---
    print("[+] Conectando con la nube de Weights & Biases...")
    run = wandb.init(
        project="Tesis-IHC-EsSalud",
        name="upload-deepliif-segmenter",
        job_type="artifact-upload"
    )

    # --- 3. CREAR EL ARTEFACTO DE MLOps ---
    print(f"[+] Registrando el modelo binario. Tamaño: {os.path.getsize(MODEL_PATH) // 1024} KB")
    artifact = wandb.Artifact(
        name="deepliif-segmenter",
        type="model",
        description="Pesos oficiales del segmentador de instancias (G51) de DeepLIIF congelados para la Fase 2.",
        metadata={
            "source": "DeepLIIF Official G51",
            "framework_target": "Segmentation Net",
            "status": "frozen"
        }
    )

    # Adjuntar el archivo físico al paquete del artefacto
    artifact.add_file(MODEL_PATH)

    # --- 4. SUBIR Y CONGELAR ---
    print("[+] Subiendo el archivo a tu repositorio central en WandB... (Por favor, espera)")
    run.log_artifact(artifact)

    # Finalizar el proceso limpiamente
    run.finish()
    print("[+] ¡Paso 1 Completado Exitosamente! El modelo base ya es un artefacto seguro y auditable.")


if __name__ == "__main__":
    upload_segmenter_artifact()