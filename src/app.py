import os
import uuid
import shutil
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

# Configurar CORS para que tu frontend en Next.js (ej. http://localhost:3000) pueda conectarse sin bloqueos
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

# Montar la carpeta estática para que los archivos sean accesibles por HTTP (ej. http://localhost:8000/static/...)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Inicializar el pipeline (carga los modelos a la GPU una sola vez al encender el servidor)
print("[+] Inicializando Pipeline de Inteligencia Artificial en el Backend...")
pipeline = InferencePipeline()


import io
import base64

@app.post("/api/quantify")
async def quantify_tissue(file: UploadFile = File(...)):
    """
    ENDPOINT CLÍNICO OPTIMIZADO: Devuelve las imágenes codificadas en Base64
    para evadir de forma definitiva los bloqueos de contenido mixto y advertencias de Ngrok.
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado.")

    request_id = str(uuid.uuid4())[:8]
    temp_input_path = os.path.join(UPLOAD_DIR, f"input_{request_id}_{file.filename}")

    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Inferencia en GPU
        he_t, fake_ihc_t, mask_t = pipeline.run_inference(temp_input_path)
        report = pipeline.analyze_cells(fake_ihc_t, mask_t)

        fake_ihc_rgb = pipeline.denormalize_to_numpy(fake_ihc_t)
        fake_ihc_img = Image.fromarray(fake_ihc_rgb)
        audit_img = Image.fromarray(report["audit_image_rgb"])

        # --- TRUCO INGENIERIL: CONVERSIÓN A BASE64 EN MEMORIA ---
        def convert_to_base64(pil_image):
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_str}"

        base64_ihc = convert_to_base64(fake_ihc_img)
        base64_audit = convert_to_base64(audit_img)

        # Guardado local de respaldo para auditoría física de archivos
        fake_ihc_img.save(os.path.join(OUTPUT_DIR, f"fake_ihc_{request_id}.png"))
        audit_img.save(os.path.join(OUTPUT_DIR, f"audit_{request_id}.png"))

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
                "synthetic_ihc_url": base64_ihc,  # <-- Ahora viaja la imagen incrustada aquí
                "audit_canvas_url": base64_audit   # <-- Ahora viaja la imagen incrustada aquí
            }
        }

    except Exception as e:
        print(f"[-] Error crítico en el endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    """Ruta de control para verificar que el backend responda"""
    return {"status": "online", "gpu_available": torch.cuda.is_available()}