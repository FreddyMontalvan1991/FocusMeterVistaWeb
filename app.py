import streamlit as st
import cv2
import threading
from ultralytics import YOLO
import time
from pymongo import MongoClient
from datetime import datetime

# =============================
# CONFIGURACIÓN
# =============================
MODEL_PATH = "app/extras/best.pt"
MONGO_URI = "mongodb://localhost:27017/" 
DB_NAME = "focusmeter_db"
COLLECTION_NAME = "atencion_logs"

# =============================
# PROCESADOR PERSISTENTE
# =============================
class BackgroundMonitor:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        # Resolución moderada para balancear calidad/CPU
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.model = YOLO(MODEL_PATH)
        
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
        except:
            self.collection = None

        self.display_frame = None  # El frame que el usuario verá
        self.nivel_atencion = 0
        self.is_running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self.update_loop, daemon=True)
        self.thread.start()

    def update_loop(self):
        frame_count = 0
        # Reducimos el salto a 5 para que el análisis sea más frecuente pero aún ligero
        SKIP_FRAMES = 5 
        last_db_save = time.time()
        
        # Guardamos el último resultado de YOLO para persistirlo en los frames intermedios
        last_annotated_results = None

    def update_loop(self):
        frame_count = 0
        SKIP_FRAMES = 5 
        last_db_save = time.time()
        
        while self.is_running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue

            frame_count += 1
            
            # --- LÓGICA DE INFERENCIA ---
            if frame_count >= SKIP_FRAMES:
                # Realizamos el análisis
                results = self.model(frame, conf=0.5, verbose=False, imgsz=320)
                
                boxes = results[0].boxes
                total = len(boxes)
                atentos = sum(1 for b in boxes if self.model.names[int(b.cls[0])].lower() in ["atento", "attentive"])
                
                with self.lock:
                    self.nivel_atencion = atentos / total if total > 0 else 0
                    # Generamos el frame con los cuadros de YOLO
                    self.display_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                
                frame_count = 0
                
                # Guardado en DB
                if time.time() - last_db_save > 10:
                    self.save_to_mongo()
                    last_db_save = time.time()
            else:
                # OPTIMIZACIÓN CLAVE: 
                # Si no toca analizar, solo actualizamos 'display_frame' si está vacío
                # Opcionalmente podrías actualizarlo siempre para fluidez, 
                # pero el análisis solo se verá cuando frame_count == SKIP_FRAMES.
                with self.lock:
                    if self.display_frame is None:
                        self.display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Control de FPS
            time.sleep(max(0, 0.033 - (time.time() - start_time)))

    def save_to_mongo(self):
        if self.collection is not None:
            try:
                self.collection.insert_one({
                    "timestamp": datetime.now(),
                    "nivel_atencion": self.nivel_atencion
                })
            except: pass

# =============================
# INICIALIZACIÓN
# =============================
@st.cache_resource
def get_persistent_monitor():
    for idx in [2, 0, 1]:
        test_cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if test_cap.isOpened():
            test_cap.release()
            return BackgroundMonitor(idx)
    return None

# =============================
# INTERFAZ STREAMLIT
# =============================
st.set_page_config(page_title="FocusMeter PFC", layout="wide")
st.title("📹 Monitoreo Estudiantil en Tiempo Real")

monitor = get_persistent_monitor()

if "view_active" not in st.session_state:
    st.session_state.view_active = False

col1, col2 = st.columns(2)
if col1.button("▶️ Ver Monitoreo"):
    st.session_state.view_active = True
if col2.button("⏹️ Ocultar"):
    st.session_state.view_active = False

frame_placeholder = st.empty()
metric_placeholder = st.empty()

if st.session_state.view_active:
    while st.session_state.view_active:
        with monitor.lock:
            img = monitor.display_frame
            nivel = monitor.nivel_atencion

        if img is not None:
            # Mostramos el frame que contiene los cuadros (si fue el de análisis)
            frame_placeholder.image(img, use_container_width=True)
            
            if nivel >= 0.7: metric_placeholder.success(f"🟢 Atención Alta: {nivel:.0%}")
            elif nivel >= 0.4: metric_placeholder.warning(f"🟡 Atención Media: {nivel:.0%}")
            else: metric_placeholder.error(f"🔴 Atención Baja: {nivel:.0%}")
        
        time.sleep(0.04)
else:
    st.info("🛰️ El servidor está procesando y guardando datos en segundo plano.")