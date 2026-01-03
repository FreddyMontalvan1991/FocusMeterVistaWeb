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
MONGO_URI = "mongodb://localhost:27017/"  # Ajusta tu URI aquí
DB_NAME = "focusmeter_db"
COLLECTION_NAME = "atencion_logs"

# =============================
# PROCESADOR PERSISTENTE
# =============================
class BackgroundMonitor:
    def __init__(self, index):
        # Configuración de Cámara en Ubuntu (V4L2)
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        # Bajamos resolución para optimizar CPU (Crucial)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        # Carga de Modelo YOLO
        self.model = YOLO(MODEL_PATH)
        
        # Conexión MongoDB
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
        except:
            self.collection = None

        # Variables de estado
        self.frame = None
        self.nivel_atencion = 0
        self.is_running = True
        self.lock = threading.Lock()
        
        # Hilo de procesamiento
        self.thread = threading.Thread(target=self.update_loop, daemon=True)
        self.thread.start()

    def update_loop(self):
        frame_count = 0
        # OPTIMIZACIÓN: Analizar solo 1 de cada 10 cuadros (aprox. 2-3 veces por segundo)
        # Esto reduce el uso de CPU en un 80-90%
        SKIP_FRAMES = 10 
        
        # Temporizador para base de datos (guardar cada 10 segundos para no saturar)
        last_db_save = time.time()

        while self.is_running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue

            frame_count += 1
            
            # --- LÓGICA DE INFERENCIA SELECTIVA ---
            if frame_count >= SKIP_FRAMES:
                # Inferencia ligera (imgsz=320 para menos carga en CPU)
                results = self.model(frame, conf=0.5, verbose=False, imgsz=320)
                
                boxes = results[0].boxes
                total = len(boxes)
                atentos = sum(1 for b in boxes if self.model.names[int(b.cls[0])].lower() in ["atento", "attentive"])
                
                with self.lock:
                    self.nivel_atencion = atentos / total if total > 0 else 0
                    # Generamos la imagen con cuadros solo cuando toca analizar
                    self.frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                
                frame_count = 0
                
                # --- GUARDAR EN BASE DE DATOS (Cada 10 seg) ---
                if time.time() - last_db_save > 10:
                    self.save_to_mongo()
                    last_db_save = time.time()
            else:
                # Frames intermedios: Solo actualizamos la imagen base (sin YOLO)
                with self.lock:
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Control de FPS del hilo (aprox 30 FPS de lectura)
            time_to_sleep = max(0, 0.033 - (time.time() - start_time))
            time.sleep(time_to_sleep)

    def save_to_mongo(self):
        if self.collection is not None:
            data = {
                "timestamp": datetime.now(),
                "nivel_atencion": self.nivel_atencion
            }
            try:
                self.collection.insert_one(data)
            except:
                pass # Evita que el programa caiga si falla la BD

# =============================
# INICIALIZACIÓN SINGLETON
# =============================
@st.cache_resource
def get_persistent_monitor():
    # Probamos índices de cámara comunes en Ubuntu
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
st.title("📹 Monitoreo Estudiantil Persistente")

monitor = get_persistent_monitor()

if monitor is None:
    st.error("❌ No se detectó cámara en /dev/video0 o /dev/video2")
    st.stop()

# Manejo de estado de la vista
if "view_active" not in st.session_state:
    st.session_state.view_active = False

# Sidebar de información constante
st.sidebar.header("⚙️ Estado del Sistema")
st.sidebar.success("Servidor: Activo")
st.sidebar.info("Análisis en 2do plano: CORRIENDO")

# Botones de control de vista
col1, col2 = st.columns(2)
if col1.button("▶️ Ver Cámara"):
    st.session_state.view_active = True
if col2.button("⏹️ Ocultar Cámara"):
    st.session_state.view_active = False

# Contenedores visuales
frame_placeholder = st.empty()
metric_placeholder = st.empty()

# --- BUCLE DE VISUALIZACIÓN ---
if st.session_state.view_active:
    # Este bucle solo corre si el usuario quiere ver la cámara
    while st.session_state.view_active:
        with monitor.lock:
            img = monitor.frame
            nivel = monitor.nivel_atencion

        if img is not None:
            frame_placeholder.image(img, use_container_width=True)
            
            # Actualizar Semáforo/Métrica
            if nivel >= 0.7:
                metric_placeholder.success(f"🟢 Nivel de Atención: {nivel:.0%}")
            elif nivel >= 0.4:
                metric_placeholder.warning(f"🟡 Nivel de Atención: {nivel:.0%}")
            else:
                metric_placeholder.error(f"🔴 Nivel de Atención: {nivel:.0%}")
        
        time.sleep(0.05) # Actualización web a 20 FPS
else:
    frame_placeholder.info("📺 La vista está desactivada, pero el servidor sigue analizando y guardando datos en MongoDB.")
    # Mostrar métrica estática aunque la cámara no se vea
    with monitor.lock:
        st.metric("Último Nivel Registrado", f"{monitor.nivel_atencion:.0%}")