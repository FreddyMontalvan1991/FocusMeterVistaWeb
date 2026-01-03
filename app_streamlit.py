import streamlit as st
import requests
import time

st.set_page_config(page_title="Monitor Web", layout="wide")

st.title("📹 Monitoreo en Tiempo Real")

# Sidebar para navegar
pagina = st.sidebar.selectbox("Ir a:", ["Monitor Principal", "Estadísticas", "Configuración"])

if pagina == "Monitor Principal":
    col1, col2 = st.columns([3, 1])
    
    with col1:
        placeholder_img = st.empty()
    
    with col2:
        st.subheader("Estado de Atención")
        semaforo = st.empty()

    # Botón para activar el visor (no el modelo, solo el visor)
    if st.toggle("Ver transmisión"):
        while True:
            try:
                # 1. Obtener imagen del servidor FastAPI
                res_img = requests.get("http://localhost:8000/video", timeout=1)
                # 2. Obtener datos del semáforo
                res_data = requests.get("http://localhost:8000/datos", timeout=1).json()

                if res_img.status_code == 200:
                    placeholder_img.image(res_img.content)
                
                # Actualizar Semáforo
                nivel = res_data['nivel']
                if nivel >= 0.7: semaforo.success(f"🟢 Alta ({nivel:.0%})")
                elif nivel >= 0.4: semaforo.warning(f"🟡 Media ({nivel:.0%})")
                else: semaforo.error(f"🔴 Baja ({nivel:.0%})")
                
                time.sleep(0.03) # Control de FPS para no saturar el navegador
            except:
                st.error("Servidor de IA desconectado.")
                break

elif pagina == "Estadísticas":
    st.write("Aquí puedes poner gráficas sin que el modelo se detenga.")