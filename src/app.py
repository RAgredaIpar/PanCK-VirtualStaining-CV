import os
import uuid
import shutil
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

# Importamos tu pipeline calibrado con el récord de Pearson (0.4255)
from pipeline_quantifier import InferencePipeline

app = FastAPI(
    title="API de Cuantificación Digital - EsSalud",
    description="Backend de IA para la síntesis cromática e inmunohistoquímica automatizada."
)

# Configurar CORS para que tu frontend en Next.js pueda conectarse sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción cambia esto por la URL de tu Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolución de rutas y creación de directorios para archivos estáticos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Montar la carpeta estática para respaldos físicos
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Inicializar el pipeline (carga los modelos a la GPU una sola vez al encender el servidor)
print("[+] Inicializando Pipeline de Inteligencia Artificial en el Backend...")
pipeline = InferencePipeline()


@app.post("/api/quantify")
async def quantify_tissue(file: UploadFile = File(...)):
    """
    ENDPOINT CLÍNICO OPTIMIZADO: Devuelve las imágenes codificadas en Base64
    e incluye el mapa de interpretabilidad Score-CAM (IA Explicable).
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado.")

    request_id = str(uuid.uuid4())[:8]
    temp_input_path = os.path.join(UPLOAD_DIR, f"input_{request_id}_{file.filename}")

    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Inferencia base en GPU y Motor Analítico HSV
        he_t, fake_ihc_t, mask_t = pipeline.run_inference(temp_input_path)
        report = pipeline.analyze_cells(fake_ihc_t, mask_t)

        # 2. NUEVO: Ejecutar el módulo de IA Explicable (Score-CAM)
        heatmap_rgb = pipeline.generate_score_cam(temp_input_path)

        # 3. Convertir tensores y matrices a objetos de imagen PIL
        fake_ihc_rgb = pipeline.denormalize_to_numpy(fake_ihc_t)
        fake_ihc_img = Image.fromarray(fake_ihc_rgb)
        audit_img = Image.fromarray(report["audit_image_rgb"])
        heatmap_img = Image.fromarray(heatmap_rgb)  # <-- Nueva imagen PIL para el mapa de calor

        # --- TRUCO INGENIERIL: CONVERSIÓN A BASE64 EN MEMORIA ---
        def convert_to_base64(pil_image):
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_str}"

        base64_ihc = convert_to_base64(fake_ihc_img)
        base64_audit = convert_to_base64(audit_img)
        base64_heatmap = convert_to_base64(heatmap_img)  # <-- Codificación Base64 de la atención de la IA

        # 4. Guardado local de respaldo para auditoría física de archivos e historial
        fake_ihc_img.save(os.path.join(OUTPUT_DIR, f"fake_ihc_{request_id}.png"))
        audit_img.save(os.path.join(OUTPUT_DIR, f"audit_{request_id}.png"))
        heatmap_img.save(os.path.join(OUTPUT_DIR, f"heatmap_{request_id}.png"))  # <-- Respaldo físico del mapa

        pos_idx = report["positivity_index"]
        if pos_idx < 15.0:
            risk_level = "LOW"
            risk_color = "#10B981"
            risk_desc = "Baja densidad de inmunopositividad Pan-CK. Predominio de estroma o tejido sano."
        elif 15.0 <= pos_idx <= 40.0:
            risk_level = "MODERATE"
            risk_color = "#F59E0B"
            risk_desc = "Presencia moderada de proliferación epitelial tumoral. Requiere correlación clínica."
        else:
            risk_level = "HIGH"
            risk_color = "#EF4444"
            risk_desc = "Alta densidad celular inmunopositiva detectada. Sugiere invasión tumoral activa masiva."

        # 5. Retornar Payload ampliado incluyendo Score-CAM
        return {
            "status": "success",
            "metadata": {
                "request_id": request_id,
                "original_filename": file.filename
            },
            "analytics": {
                "total_nuclei_detected": report["total_nuclei"],
                "positive_nuclei_count": report["positive_nuclei"],
                "negative_nuclei_count": report["negative_nuclei"],
                "positivity_index_percentage": pos_idx
            },
            "clinical_risk": {
                "level": risk_level,
                "color_code": risk_color,
                "description": risk_desc
            },
            "visual_payloads": {
                "synthetic_ihc_url": base64_ihc,
                "audit_canvas_url": base64_audit,
                "score_cam_url": base64_heatmap  # <-- AQUÍ ENVIAMOS EL MAPA DE CALOR A NEXT.JS
            }
        }

    except Exception as e:
        print(f"[-] Error crítico en el endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    """Ruta de control para verificar que el backend responda"""
    return {"status": "online", "gpu_available": torch.cuda.is_available()}