import streamlit as st
import cv2
import threading
from ultralytics import YOLO
import time

# =============================
# CONFIGURACIÓN
# =============================
MODEL_PATH = "app/extras/best.pt"

class VideoProcessor:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.model = YOLO(MODEL_PATH)
        self.frame = None
        self.nivel_atencion = 0
        self.running = False  # Controla si el hilo procesa frames
        self.lock = threading.Lock()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Inferencia YOLO
            results = self.model(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()

            # Lógica de atención
            atentos = 0
            total = len(results[0].boxes)
            for box in results[0].boxes:
                if self.model.names[int(box.cls[0])].lower() in ["atento", "attentive"]:
                    atentos += 1
            
            with self.lock:
                self.frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                self.nivel_atencion = atentos / total if total > 0 else 0

# =============================
# GESTIÓN DE ESTADO (CACHE)
# =============================
@st.cache_resource
def get_processor():
    # Intentar índices comunes en Ubuntu
    for idx in [2, 0]:
        test_cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if test_cap.isOpened():
            test_cap.release()
            return VideoProcessor(idx)
    return None

processor = get_processor()

# =============================
# INTERFAZ DE USUARIO
# =============================
st.title("📹 Monitoreo Controlado por Hilos")

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

col1, col2 = st.columns(2)

if col1.button("▶️ Iniciar monitoreo"):
    st.session_state.monitoring = True
    processor.start()

if col2.button("⏹️ Detener monitoreo"):
    st.session_state.monitoring = False
    processor.stop()

# Áreas de visualización
frame_window = st.image([])
semaforo = st.empty()

# =============================
# BUCLE DE VISTA (LOOP PRINCIPAL)
# =============================
if st.session_state.monitoring:
    while st.session_state.monitoring:
        with processor.lock:
            img = processor.frame
            nivel = processor.nivel_atencion

        if img is not None:
            frame_window.image(img, use_container_width=True)
            
            # Actualizar Semáforo
            if nivel >= 0.7:
                semaforo.success(f"🟢 Nivel de Atención: {nivel:.0%}")
            elif nivel >= 0.4:
                semaforo.warning(f"🟡 Nivel de Atención: {nivel:.0%}")
            else:
                semaforo.error(f"🔴 Nivel de Atención: {nivel:.0%}")
        
        time.sleep(0.01) # Evita sobrecarga de CPU en el navegador
else:
    st.info("El sistema está en espera. Presiona 'Iniciar' para ver la cámara.")