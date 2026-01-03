import cv2
from fastapi import FastAPI, Response
from ultralytics import YOLO
import threading
import time

app = FastAPI()

# --- CONFIGURACIÓN ---
MODEL_PATH = "app/extras/best.pt"
model = YOLO(MODEL_PATH)

# Variables Globales en RAM
ultima_imagen = None
datos_estado = {"nivel": 0, "total": 0, "camara": "Buscando..."}

def detectar_mejor_camara():
    """
    Escanea índices del 1 al 5 para USB, y cae al 0 para integrada.
    """
    print("🔍 Escaneando puertos de cámara...")
    # Intentar cámaras externas primero (índices 1 al 5)
    for index in range(1, 6):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cap.release()
                print(f"✅ Cámara USB detectada en el índice: {index}")
                return index
            cap.release()
    
    # Si no, intentar la integrada (0)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("💻 No se detectó USB. Usando cámara integrada (índice 0).")
        cap.release()
        return 0
    
    print("❌ No se encontró ninguna cámara disponible.")
    return None

def bucle_deteccion():
    global ultima_imagen, datos_estado
    
    idx = detectar_mejor_camara()
    if idx is None:
        datos_estado["status"] = "Error: No hay cámara"
        return

    cap = cv2.VideoCapture(idx)
    # Ajustes de fluidez
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    datos_estado["camara"] = f"Puerto {idx}"

    print("🚀 Modelo iniciado y procesando...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Error al leer fotograma. Reintentando...")
            time.sleep(1)
            continue

        # Inferencia
        results = model(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes
        atentos = 0
        total = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            
            # Lógica de color según tu modelo
            if label.lower() in ["atento", "attentive"]:
                color, atentos = (0, 255, 0), atentos + 1
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Actualizar datos en RAM
        nivel = atentos / total if total > 0 else 0
        datos_estado["nivel"] = nivel
        datos_estado["total"] = total

        # Mostrar progreso en la consola del servidor (una sola línea)
        print(f"Detectando en {datos_estado['camara']} | Personas: {total} | Nivel: {nivel:.2%}", end="\r")

        # Convertir a bytes para Streamlit
        _, buffer = cv2.imencode('.jpg', frame)
        ultima_imagen = buffer.tobytes()

# Iniciar el hilo del modelo automáticamente al arrancar el servidor
threading.Thread(target=bucle_deteccion, daemon=True).start()

@app.get("/video")
async def get_video():
    if ultima_imagen is None:
        return Response(status_code=404)
    return Response(content=ultima_imagen, media_type="image/jpeg")

@app.get("/datos")
async def get_datos():
    return datos_estado

if __name__ == "__main__":
    import uvicorn
    # Iniciamos el servidor
    uvicorn.run(app, host="0.0.0.0", port=8000)