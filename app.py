import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys, time
import io
import textwrap
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from ecg_pipeline import (
    download_record, load_record, apply_filters,
    detect_r_peaks, calculate_features, estimate_qrs_width,
    diagnose_with_ollama, design_notch_filter, design_bandpass_filter,
    generate_synthetic_ecg, load_custom_file
)
import scipy.signal as scipy_signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Analizador ECG — Laplace + IA",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Outfit', sans-serif !important;
        background-color: #0a0a0a !important;
        color: #e0e0e0 !important;
    }
    
    .main { background-color: #0a0a0a; }
    
    section[data-testid="stSidebar"] { 
        background-color: #111111 !important; 
        border-right: 1px solid #222222;
    }
    
    .metric-card {
        background: radial-gradient(circle at top right, rgba(0,204,255,0.03), transparent 70%), #141414;
        border: 1px solid #262626;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3a3a3a;
    }
    .metric-value { font-size: 30px; font-weight: 700; color: #00ccff; margin: 4px 0; }
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
    .metric-sub { font-size: 11px; color: #666; margin-top: 2px; }
    
    .diag-box {
        background: linear-gradient(135deg, rgba(13,31,13,0.9), rgba(8,20,8,0.9));
        border: 1px solid #1a4a1a;
        border-radius: 12px;
        padding: 22px;
        font-size: 15px;
        line-height: 1.7;
        color: #a8d8a8;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    
    .warn-box {
        background: linear-gradient(135deg, rgba(31,13,13,0.9), rgba(20,8,8,0.9));
        border: 1px solid #4a1a1a;
        border-radius: 12px;
        padding: 22px;
        font-size: 15px;
        line-height: 1.7;
        color: #d8a8a8;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0077ff, #0044cc);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0,119,255,0.2);
        transition: all 0.2s ease;
    }
    .stButton > button:hover { 
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,119,255,0.4);
        opacity: 0.95;
    }
    .stButton > button:active {
        transform: translateY(1px);
    }
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
    .stSelectbox label, .stSlider label, .stRadio label { color: #bbbbbb !important; font-weight: 500 !important; }
    hr { border-color: #222; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# IMPORTS ADICIONALES Y AUXILIARES
# ─────────────────────────────────────────────

import scipy.signal as sp

def plot_pole_zero_plotly(b, a, title="Mapa de Polos y Ceros"):
    # raíces de polinomios
    zeros = np.roots(b)
    poles = np.roots(a)
    
    # Círculo unitario
    theta = np.linspace(0, 2*np.pi, 200)
    uc_x = np.cos(theta)
    uc_y = np.sin(theta)
    
    fig = go.Figure()
    
    # Círculo unitario
    fig.add_trace(go.Scatter(
        x=uc_x, y=uc_y, mode='lines', name='Círculo Unitario',
        line=dict(color='#444444', dash='dash', width=1.5), showlegend=False
    ))
    
    # Ejes
    fig.add_shape(type="line", x0=-1.5, y0=0, x1=1.5, y1=0, line=dict(color="#222222", width=1))
    fig.add_shape(type="line", x0=0, y0=-1.5, x1=0, y1=1.5, line=dict(color="#222222", width=1))
    
    # Graficar ceros (círculo verde)
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(zeros), y=np.imag(zeros),
            mode='markers', name='Ceros (o)',
            marker=dict(color='#00ff88', size=10, symbol='circle-open', line=dict(width=2)),
            hovertemplate="Cero: %{x:.3f} + %{y:.3f}j<extra></extra>"
        ))
    
    # Graficar polos (x roja)
    if len(poles) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(poles), y=np.imag(poles),
            mode='markers', name='Polos (x)',
            marker=dict(color='#ff4444', size=10, symbol='x', line=dict(width=2)),
            hovertemplate="Polo: %{x:.3f} + %{y:.3f}j<extra></extra>"
        ))
        
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=13)),
        width=340,
        height=340,
        paper_bgcolor='#0f0f0f',
        plot_bgcolor='#0f0f0f',
        xaxis=dict(range=[-1.5, 1.5], scaleanchor="y", scaleratio=1, gridcolor='#1e1e1e', title="Real"),
        yaxis=dict(range=[-1.5, 1.5], gridcolor='#1e1e1e', title="Imaginario"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(10,10,10,0.8)', bordercolor='#222', borderwidth=1, font=dict(size=9))
    )
    return fig


def build_html_report(results):
    """Crea un reporte HTML descargable con las métricas principales."""
    features = results['features']
    diagnosis = results.get('diagnosis') or "No disponible."
    record = results.get('record', 'sin registro')
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    has_anom = features['anomalous_beats'] > 2 or features['hr_mean'] > 100 or features['hr_mean'] < 50
    conclusion = (
        "La lectura requiere revisión: se encontraron valores fuera del rango esperado o varios latidos irregulares."
        if has_anom else
        "La lectura general es estable en los indicadores principales analizados."
    )

    return f"""<!doctype html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>Reporte ECG</title>
    <style>
        body {{ font-family: Arial, sans-serif; color: #202124; margin: 40px; line-height: 1.5; }}
        h1 {{ margin-bottom: 4px; }}
        .muted {{ color: #666; font-size: 13px; }}
        .box {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 18px 0; }}
        .conclusion {{ background: #f5f8fb; border-color: #cfd8e3; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 9px; text-align: left; }}
        th {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Reporte de analisis ECG</h1>
    <div class="muted">Generado: {generated_at} | Registro: {record} | Duracion: {features['duration_s']} segundos</div>

    <div class="box conclusion">
        <h2>Conclusion rapida</h2>
        <p>{conclusion}</p>
    </div>

    <div class="box">
        <h2>Metricas principales</h2>
        <table>
            <tr><th>Indicador</th><th>Valor</th><th>Lectura</th></tr>
            <tr><td>Frecuencia cardiaca promedio</td><td>{features['hr_mean']} bpm</td><td>Rango de referencia: 60 a 100 bpm</td></tr>
            <tr><td>Frecuencia minima-maxima</td><td>{features['hr_min']} - {features['hr_max']} bpm</td><td>Variacion durante la muestra</td></tr>
            <tr><td>RMSSD</td><td>{features['rmssd']} ms</td><td>Indicador de variabilidad entre latidos</td></tr>
            <tr><td>pNN50</td><td>{features['pnn50']}%</td><td>Porcentaje de cambios RR mayores a 50 ms</td></tr>
            <tr><td>QRS estimado</td><td>{features['qrs_width_ms']} ms</td><td>Referencia habitual: 80 a 120 ms</td></tr>
            <tr><td>Calidad de senal</td><td>{features['snr_db']} dB</td><td>Relacion senal/ruido</td></tr>
            <tr><td>Latidos detectados</td><td>{features['total_beats']}</td><td>Total dentro de la muestra</td></tr>
            <tr><td>Latidos irregulares</td><td>{features['anomalous_beats']}</td><td>Detectados por variacion RR</td></tr>
        </table>
    </div>

    <div class="box">
        <h2>Interpretacion automatica</h2>
        <p>{diagnosis}</p>
    </div>

    <p class="muted">
        Este reporte es educativo y orientativo. No reemplaza una valoracion medica profesional.
    </p>
</body>
</html>"""


def build_pdf_report(results):
    """Crea un PDF descargable con resumen, métricas y gráficas principales."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    features = results['features']
    diagnosis = results.get('diagnosis') or "No disponible."
    record = results.get('record', 'sin registro')
    fs = results['fs']
    t = results['t']
    ecg_clean = results['ecg_clean']
    peaks = results['peaks']
    anomalous_peaks = np.asarray(results['anomalous_peaks'])
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    has_anom = features['anomalous_beats'] > 2 or features['hr_mean'] > 100 or features['hr_mean'] < 50
    conclusion = (
        "La lectura requiere revision porque se encontraron valores fuera del rango esperado o varios latidos irregulares."
        if has_anom else
        "La lectura general es estable en los indicadores principales analizados."
    )

    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.94, "Reporte de analisis ECG", fontsize=20, fontweight="bold")
        fig.text(0.08, 0.91, f"Generado: {generated_at} | Registro: {record} | Duracion: {features['duration_s']} segundos", fontsize=9, color="#555555")
        fig.text(0.08, 0.85, "Conclusion rapida", fontsize=14, fontweight="bold")
        fig.text(0.08, 0.82, "\n".join(textwrap.wrap(conclusion, width=92)), fontsize=11)

        fig.text(0.08, 0.75, "Lectura en lenguaje natural", fontsize=14, fontweight="bold")
        natural_text = (
            f"La muestra analizada contiene {features['total_beats']} latidos detectados. "
            f"La frecuencia cardiaca promedio fue de {features['hr_mean']} bpm, con un rango entre "
            f"{features['hr_min']} y {features['hr_max']} bpm. Se encontraron "
            f"{features['anomalous_beats']} latidos con variacion irregular segun los intervalos RR. "
            f"La calidad estimada de la senal fue de {features['snr_db']} dB."
        )
        fig.text(0.08, 0.70, "\n".join(textwrap.wrap(natural_text, width=92)), fontsize=10)

        metrics = [
            ["Frecuencia cardiaca", f"{features['hr_mean']} bpm", "60 a 100 bpm"],
            ["Frecuencia min-max", f"{features['hr_min']} - {features['hr_max']} bpm", "Estable"],
            ["RMSSD", f"{features['rmssd']} ms", "> 20 ms"],
            ["pNN50", f"{features['pnn50']}%", "> 10%"],
            ["QRS estimado", f"{features['qrs_width_ms']} ms", "80 a 120 ms"],
            ["Calidad de senal", f"{features['snr_db']} dB", "> 10 dB"],
            ["Latidos detectados", str(features['total_beats']), "Segun duracion"],
            ["Latidos irregulares", str(features['anomalous_beats']), "Ideal: 0"],
        ]
        ax_table = fig.add_axes([0.08, 0.30, 0.84, 0.32])
        ax_table.axis("off")
        table = ax_table.table(
            cellText=metrics,
            colLabels=["Indicador", "Valor", "Referencia"],
            cellLoc="left",
            colLoc="left",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.45)

        fig.text(0.08, 0.22, "Interpretacion automatica", fontsize=14, fontweight="bold")
        fig.text(0.08, 0.13, "\n".join(textwrap.wrap(diagnosis, width=92))[:900], fontsize=9)
        fig.text(0.08, 0.06, "Reporte educativo y orientativo. No reemplaza una valoracion medica profesional.", fontsize=8, color="#666666")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(t, ecg_clean, color="#0077cc", linewidth=1.0, label="ECG filtrado")
        peaks_in = peaks[peaks < len(t)]
        if len(peaks_in) > 0:
            ax.scatter(t[peaks_in], ecg_clean[peaks_in], color="#009966", s=20, label="Latidos detectados")
        anom_in = anomalous_peaks[anomalous_peaks < len(t)]
        if len(anom_in) > 0:
            ax.scatter(t[anom_in], ecg_clean[anom_in], color="#cc3333", s=45, marker="x", label="Latidos irregulares")
        ax.set_title("ECG filtrado con latidos detectados")
        ax.set_xlabel("Tiempo (segundos)")
        ax.set_ylabel("Amplitud normalizada")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        rr_intervals_ms = np.diff(peaks) / fs * 1000
        if len(rr_intervals_ms) > 0:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            rr_times = peaks[1:] / fs
            ax.plot(rr_times, rr_intervals_ms, color="#cc8800", marker="o", linewidth=1.4)
            ax.axhline(float(np.mean(rr_intervals_ms)), color="#0077cc", linestyle="--", label="Promedio RR")
            ax.set_title("Regularidad entre latidos")
            ax.set_xlabel("Tiempo (segundos)")
            ax.set_ylabel("Intervalo RR (ms)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
            fig.text(0.08, 0.01, "Una linea estable sugiere ritmo mas regular. Saltos grandes indican variacion entre latidos.", fontsize=9)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ─────────────────────────────────────────────
# SIDEBAR / PANEL DE CONTROL
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Analizador de ECG")
    st.markdown("---")

    st.markdown("### Fuente de la prueba")

    # Valores técnicos por defecto (no visibles al usuario)
    notch_freq = 60
    notch_q = 30
    bp_low = 0.5
    bp_high = 40
    sim_hr = 75
    sim_powerline = 0.2
    sim_muscle = 0.1
    sim_drift = 0.3
    uploaded = None
    fs = 360
    use_ollama = True
    ollama_model = "llama3.1:8b"
    ollama_host = "http://127.0.0.1:11434"

    data_source = st.radio(
        "Fuente:",
        [
            "Demostración (generada automáticamente)",
            "Base de datos clínica (MIT-BIH)",
            "Mi propio archivo"
        ],
        label_visibility="collapsed"
    )

    if data_source == "Demostración (generada automáticamente)":
        record_name = "synthetic"
        st.markdown("""
        <p style='font-size:13px;color:#aaa;line-height:1.6;margin-top:8px;'>
        Genera una señal de corazón de ejemplo para explorar la herramienta sin necesidad de archivos.
        </p>""", unsafe_allow_html=True)

    elif data_source == "Base de datos clínica (MIT-BIH)":
        record_options = {
            "Ritmo normal": "100",
            "Latidos irregulares": "101",
            "Ritmo muy rápido (taquicardia)": "200",
            "Contracciones prematuras": "201",
            "Bloqueo de conducción cardíaca": "202",
        }
        st.markdown("Tipo de registro:", unsafe_allow_html=False)
        record_label = st.selectbox(
            "Registro:",
            list(record_options.keys()),
            label_visibility="collapsed"
        )
        record_name = record_options[record_label]

    else:
        record_name = "custom"
        st.markdown("""
        <p style='font-size:13px;color:#aaa;line-height:1.6;margin-top:8px;'>
        Sube un archivo de texto (.csv o .txt) donde cada línea sea un número que representa el voltaje de la señal.
        </p>""", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Selecciona tu archivo:",
            type=['csv', 'txt'],
            label_visibility="collapsed"
        )
        fs = st.number_input(
            "Velocidad de captura (lecturas por segundo)",
            min_value=10, max_value=1000, value=250,
            help="Si no sabes este valor, deja 250"
        )

    st.markdown("---")
    duration_s = st.slider(
        "Segundos a analizar:",
        5, 60, 10,
        help="Cuántos segundos de señal procesar"
    )

    st.markdown("---")
    with st.expander("Configuración avanzada (solo para técnicos)"):
        st.markdown("<p style='font-size:11px;color:#555;'>Esta sección es solo para usuarios con conocimientos técnicos en procesamiento de señales.</p>", unsafe_allow_html=True)
        st.markdown("**Simulador:**")
        sim_hr = st.slider("Frecuencia cardíaca (bpm)", 50, 180, sim_hr)
        sim_powerline = st.slider("Ruido eléctrico", 0.0, 1.5, sim_powerline, 0.05)
        sim_muscle = st.slider("Ruido muscular", 0.0, 1.0, sim_muscle, 0.05)
        sim_drift = st.slider("Movimiento/Respiración", 0.0, 2.0, sim_drift, 0.1)
        st.markdown("**Filtros digitales:**")
        notch_freq = st.selectbox("Red eléctrica", [60, 50], index=0, help="60 Hz América / 50 Hz Europa")
        notch_q = st.slider("Factor Q (Notch)", 10, 60, notch_q)
        bp_low = st.slider("Corte inferior (Hz)", 0.1, 2.0, bp_low, 0.1)
        bp_high = st.slider("Corte superior (Hz)", 20, 80, bp_high, 5)
        st.markdown("**IA local (Ollama):**")
        use_ollama = st.toggle("Activar diagnóstico IA", value=use_ollama)
        ollama_model = st.selectbox("Modelo", ["llama3.1:8b", "llama3", "llama3.2", "mistral", "gemma2"])
        ollama_host = st.text_input("Servidor Ollama", ollama_host)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("# Generador y Analizador de Señales Biomédicas")
    st.markdown("**Simula, limpia e interpreta señales ECG para convertir datos crudos en una lectura clara**")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    source_label = "SIMULADOR" if data_source == "Demostración (generada automáticamente)" else ("ARCHIVO" if data_source == "Mi propio archivo" else "MIT-BIH")
    st.markdown(f"""
    <div style="background:#111;border:1px solid #222;border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:11px;color:#555">FUENTE DE SEÑAL</div>
        <div style="font-size:14px;font-weight:600;color:#00ccff">{source_label}</div>
        <div style="font-size:10px;color:#555">{fs} Hz · Fs</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# INTRODUCCIÓN SIMPLE SOBRE EL SENTIDO DE LA APP
# ─────────────────────────────────────────────
with st.expander("¿Para qué sirve esta app?", expanded=False):
    col_intro1, col_intro2 = st.columns([1, 1])
    
    with col_intro1:
        st.markdown("""
        #### El problema
        
        Un electrocardiograma no llega perfecto: puede traer ruido por movimiento,
        electricidad del ambiente o mala captura. A simple vista, eso hace difícil
        saber dónde están los latidos reales.

        Esta herramienta toma esa señal, la limpia y resume lo importante:
        el ritmo, la regularidad, la variabilidad y la calidad de la grabación.
        """)
    
    with col_intro2:
        st.markdown("""
        #### La idea matemática

        La transformada de Laplace se usa como base para diseñar filtros:
        uno elimina el ruido de la red eléctrica y otro deja pasar principalmente
        las frecuencias donde vive la información del corazón.

        En palabras simples: la matemática permite separar señal útil de ruido,
        y la app traduce ese resultado a indicadores que cualquier persona puede leer.
        """)
    
    st.divider()
    st.caption("La vista técnica conserva las fórmulas, polos, ceros y respuesta en frecuencia para sustentar la parte académica.")

st.markdown("---")

# ─────────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────────

if 'results' not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────
# BOTONES DE ACCIÓN
# ─────────────────────────────────────────────

left_btn, col_btn, right_btn = st.columns([1, 1.2, 1])
with col_btn:
    run_analysis = st.button("Iniciar Análisis Completo", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────

if run_analysis:
    ecg_seg = None
    t = None
    
    # 1. Cargar señal según la fuente
    if data_source == "Demostración (generada automáticamente)":
        ecg_raw, ecg_clean_orig, t_orig = generate_synthetic_ecg(
            duration_s=duration_s, fs=fs, hr=sim_hr,
            noise_powerline=sim_powerline, noise_muscle=sim_muscle,
            noise_drift=sim_drift, powerline_freq=notch_freq
        )
        ecg_seg = ecg_raw
        t = t_orig
        
    elif data_source == "Base de datos clínica (MIT-BIH)":
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(os.path.join(DATA_DIR, f"{record_name}.dat")) or not os.path.exists(os.path.join(DATA_DIR, f"{record_name}.hea")):
            with st.spinner(f"Descargando registro {record_name} de la base de datos clínica..."):
                try:
                    download_record(record_name, DATA_DIR)
                except Exception as e:
                    st.error(f"Error al descargar el registro: {e}")
                    st.stop()
        
        ecg_raw, fs, annotation = load_record(record_name, DATA_DIR)
        n = int(duration_s * fs)
        ecg_seg = ecg_raw[:n]
        t = np.arange(n) / fs
        
    elif data_source == "Mi propio archivo":
        if uploaded is None:
            st.warning("Por favor, selecciona y sube tu archivo primero.")
            st.stop()
        try:
            raw_data = load_custom_file(uploaded)
            n = int(duration_s * fs)
            if len(raw_data) < n:
                st.warning(f"El archivo tiene menos datos de los necesarios. Se analizaran solo {len(raw_data)} muestras.")
                n = len(raw_data)
            ecg_seg = raw_data[:n]
            t = np.arange(n) / fs
        except Exception as e:
            st.error(f"No se pudo leer el archivo. Verifica que sea un archivo de texto con un numero por linea. Error: {e}")
            st.stop()

    # 2. Filtrar y Procesar
    with st.spinner("Filtrando la señal y detectando complejos cardíacos..."):
        ecg_clean, ecg_notched = apply_filters(
            ecg_seg, fs=fs, notch_freq=notch_freq, notch_q=notch_q,
            lowcut=bp_low, highcut=bp_high
        )
        
        # Picos R y características
        peaks = detect_r_peaks(ecg_clean, fs)
        features = calculate_features(ecg_clean, peaks, fs, duration_s)
        
        if features is None:
            st.error("No se pudieron detectar latidos válidos. Intenta ajustar los filtros o subir una señal más clara.")
            st.stop()
            
        # Identificar latidos anómalos
        rr = np.diff(peaks)
        rr_mean, rr_std = np.mean(rr), np.std(rr)
        anom_mask = np.abs(rr - rr_mean) > 2 * rr_std
        anomalous_peaks = peaks[1:][anom_mask]
        
        # Filtros de coeficientes para la vista técnica
        nyq = fs / 2
        w0 = notch_freq / nyq
        b_notch, a_notch = sp.iirnotch(w0, notch_q)
        b_bp, a_bp = sp.butter(4, [bp_low/nyq, bp_high/nyq], btype='band')
        
        # Diagnóstico por IA
        diagnosis = None
        if use_ollama:
            with st.spinner("Consultando al sistema de diagnóstico IA... (puede tardar 10-20 segundos)"):
                diagnosis = diagnose_with_ollama(features, ollama_model, ollama_host)
        else:
            diagnosis = "El diagnóstico automático por IA está desactivado."
            
        # Guardar en sesión
        st.session_state.results = {
            'ecg_seg': ecg_seg, 'ecg_clean': ecg_clean,
            'peaks': peaks, 'anomalous_peaks': anomalous_peaks,
            'features': features, 'diagnosis': diagnosis,
            'fs': fs, 't': t, 'ecg_notched': ecg_notched,
            'b_notch': b_notch, 'a_notch': a_notch,
            'b_bp': b_bp, 'a_bp': a_bp,
            'record': record_name
        }

# ─────────────────────────────────────────────
# RENDERIZADO DE RESULTADOS (MULTIPESTAÑA)
# ─────────────────────────────────────────────

if st.session_state.results:
    r = st.session_state.results
    f = r['features']
    t = r['t']
    
    # Crear pestañas profesionales
    tab1, tab2, tab3 = st.tabs(["Resumen fácil", "Antes y después", "Vista técnica"])
    
    # ==========================================
    # TAB 1: REPORTE DE SALUD (PÚBLICO GENERAL)
    # ==========================================
    with tab1:
        has_anom = f['anomalous_beats'] > 2 or f['hr_mean'] > 100 or f['hr_mean'] < 50
        overall_title = "Revisar con atención" if has_anom else "Lectura general estable"
        overall_text = (
            "La señal muestra cambios que merecen revisarse: puede haber ritmo fuera de rango o varios latidos irregulares."
            if has_anom else
            "La señal analizada no muestra alertas fuertes en los indicadores principales. El resultado sirve como una lectura orientativa del ritmo."
        )
        st.markdown(f"""
        <div style="background:#141414;border:1px solid #262626;border-radius:12px;padding:20px;margin-bottom:20px;">
            <div style="font-size:12px;color:#777;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">Conclusión rápida</div>
            <div style="font-size:24px;color:#ffffff;font-weight:700;margin-top:4px;">{overall_title}</div>
            <div style="font-size:14px;color:#b8b8b8;line-height:1.6;margin-top:8px;">{overall_text}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Lo más importante")
        
        # Columnas de métricas amigables
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        # 1. Ritmo Cardíaco (FC)
        hr_val = f['hr_mean']
        hr_status = "Normal" if 60 <= hr_val <= 100 else ("Elevado (Taquicardia)" if hr_val > 100 else "Bajo (Bradicardia)")
        hr_color = "#00ff88" if 60 <= hr_val <= 100 else "#ff4444"
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Ritmo Cardíaco (FC)</div>
                <div class="metric-value" style="color:{hr_color}">{hr_val} <span style="font-size:14px;font-weight:400;color:#aaa">bpm</span></div>
                <div class="metric-sub">{hr_status} (rango: 60-100)</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 2. Regularidad del Ritmo
        anom_count = f['anomalous_beats']
        reg_status = "Ritmo Regular" if anom_count == 0 else (f"Irregular ({anom_count} anomalías)" if anom_count < 3 else f"Alta Irregularidad ({anom_count} anomalías)")
        reg_color = "#00ff88" if anom_count == 0 else ("#ffaa00" if anom_count < 3 else "#ff4444")
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Regularidad del Latido</div>
                <div class="metric-value" style="color:{reg_color}">{reg_status}</div>
                <div class="metric-sub">Pulsaciones estables en el tiempo</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. Variabilidad Cardíaca (HRV)
        # RMSSD es el indicador principal de HRV
        rmssd_val = f['rmssd']
        hrv_status = "Excelente" if rmssd_val > 40 else ("Estable / Normal" if rmssd_val >= 20 else "Baja (Estrés / Rigidez)")
        hrv_color = "#00ff88" if rmssd_val > 40 else ("#00ccff" if rmssd_val >= 20 else "#ffaa00")
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Adaptabilidad (HRV)</div>
                <div class="metric-value" style="color:{hrv_color}">{hrv_status}</div>
                <div class="metric-sub">RMSSD: {rmssd_val} ms · pNN50: {f['pnn50']}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 4. Calidad de Grabación (SNR)
        snr_val = f['snr_db']
        snr_status = "Excelente" if snr_val > 15 else ("Aceptable" if snr_val >= 5 else "Baja calidad (Movimiento)")
        snr_color = "#00ccff" if snr_val > 15 else ("#ffaa00" if snr_val >= 5 else "#ff4444")
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Calidad de Señal</div>
                <div class="metric-value" style="color:{snr_color}">{snr_status}</div>
                <div class="metric-sub">Relación señal/ruido: {snr_val} dB</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gráfica interactiva de ECG Limpio
        st.markdown("### Latidos detectados")
        
        fig_clean = go.Figure()
        fig_clean.add_trace(go.Scatter(
            x=t, y=r['ecg_clean'],
            name='Señal Cardíaca',
            line=dict(color='#00ccff', width=1.5)
        ))
        
        # Picos R (Latidos normales)
        peaks_in = r['peaks'][r['peaks'] < len(t)]
        fig_clean.add_trace(go.Scatter(
            x=t[peaks_in], y=r['ecg_clean'][peaks_in],
            mode='markers',
            name='Latido Detectado',
            marker=dict(color='#00ff88', size=7, symbol='circle')
        ))
        
        # Anomalías marcadas
        anom_in = r['anomalous_peaks'][r['anomalous_peaks'] < len(t)]
        if len(anom_in) > 0:
            fig_clean.add_trace(go.Scatter(
                x=t[anom_in], y=r['ecg_clean'][anom_in],
                mode='markers',
                name='Latido Irregular',
                marker=dict(color='#ff3333', size=11, symbol='x', line=dict(width=2))
            ))
            # Sombrear la zona del latido irregular
            for ap in anom_in:
                fig_clean.add_vrect(
                    x0=t[ap]-0.15, x1=t[ap]+0.15,
                    fillcolor="red", opacity=0.08, line_width=0
                )
                
        fig_clean.update_layout(
            height=350,
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0f0f0f',
            font=dict(color='#aaaaaa', size=11),
            legend=dict(bgcolor='rgba(17,17,17,0.8)', bordercolor='#222', borderwidth=1),
            margin=dict(l=45, r=20, t=10, b=30),
            xaxis=dict(title="Tiempo (segundos)", gridcolor='#151515', zerolinecolor='#222'),
            yaxis=dict(title="Amplitud normalizada (latido)", gridcolor='#151515', zerolinecolor='#222')
        )
        
        st.plotly_chart(fig_clean, use_container_width=True)

        rr_intervals_ms = np.diff(r['peaks']) / r['fs'] * 1000
        if len(rr_intervals_ms) > 0:
            st.markdown("### Regularidad entre latidos")
            fig_rr = go.Figure()
            rr_times = r['peaks'][1:] / r['fs']
            fig_rr.add_trace(go.Scatter(
                x=rr_times,
                y=rr_intervals_ms,
                mode='lines+markers',
                name='Intervalo RR',
                line=dict(color='#ffaa00', width=2),
                marker=dict(size=6, color='#ffaa00')
            ))
            fig_rr.add_hline(
                y=float(np.mean(rr_intervals_ms)),
                line_dash="dash",
                line_color="#00ccff",
                annotation_text="Promedio RR",
                annotation_position="top right"
            )
            fig_rr.update_layout(
                height=260,
                paper_bgcolor='#0a0a0a',
                plot_bgcolor='#0f0f0f',
                font=dict(color='#aaaaaa', size=11),
                showlegend=False,
                margin=dict(l=45, r=20, t=10, b=30),
                xaxis=dict(title="Tiempo (segundos)", gridcolor='#151515', zerolinecolor='#222'),
                yaxis=dict(title="Intervalo RR (ms)", gridcolor='#151515', zerolinecolor='#222')
            )
            st.plotly_chart(fig_rr, use_container_width=True)
            st.caption("Si esta línea se mantiene estable, el ritmo es más regular. Si tiene saltos grandes, hay variaciones entre latidos.")

        # Explicación simple del reporte para usuarios no técnicos
        st.markdown("""
        <div style="background: #141414; border: 1px solid #222; border-radius: 10px; padding: 20px; margin-top: 15px; margin-bottom: 25px;">
            <h4 style="margin-top:0;color:white;font-size:15px;font-weight:600;">Cómo leer este resultado:</h4>
            <ul style="font-size: 13px; color: #aaa; line-height: 1.6; margin-bottom: 0; padding-left: 20px;">
                <li><strong>Ritmo cardíaco</strong>: dice si el corazón va lento, normal o rápido durante la muestra.</li>
                <li><strong>Regularidad</strong>: revisa si los latidos aparecen con una separación parecida o si hay saltos extraños.</li>
                <li><strong>Adaptabilidad</strong>: muestra qué tanto varía el tiempo entre latidos; no todos los latidos sanos son idénticos.</li>
                <li><strong>Calidad de señal</strong>: indica si la lectura fue limpia o si el ruido pudo afectar el resultado.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Conclusiones e IA
        st.markdown("### Interpretación automática")
        if r['diagnosis']:
            box_class = "warn-box" if has_anom else "diag-box"
            status_text = "Atención:" if has_anom else "Resultado orientativo:"
            st.markdown(f"""
            <div class="{box_class}">
                <strong>{status_text} Registro {r['record']}</strong><br><br>
                {r['diagnosis']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("El reporte automático de IA está desactivado o el modelo local de Ollama no está disponible.")

        report_pdf = build_pdf_report(r)
        st.download_button(
            label="Descargar reporte PDF",
            data=report_pdf,
            file_name=f"reporte_ecg_{r['record']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
            
        st.caption("Descargo de responsabilidad: Esta aplicación es un sistema de soporte y análisis digital. No representa un diagnóstico médico clínico oficial. Consulte siempre a un especialista ante cualquier anomalía.")

    # ==========================================
    # TAB 2: ANÁLISIS DE SEÑAL Y FILTRADO
    # ==========================================
    with tab2:
        st.markdown("### Antes y después de limpiar la señal")
        st.markdown("Aquí se ve el sentido del proyecto: una señal ruidosa se transforma en una lectura más clara, donde los latidos pueden medirse mejor.")
        
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=t, y=r['ecg_seg'],
            name='Señal cruda con ruido',
            line=dict(color='#ff4444', width=0.8),
            opacity=0.5
        ))
        fig_compare.add_trace(go.Scatter(
            x=t, y=r['ecg_clean'],
            name='Señal filtrada',
            line=dict(color='#00ff88', width=1.3)
        ))
        
        fig_compare.update_layout(
            height=400,
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0f0f0f',
            font=dict(color='#aaaaaa', size=11),
            legend=dict(bgcolor='rgba(17,17,17,0.8)', bordercolor='#222', borderwidth=1),
            margin=dict(l=45, r=20, t=10, b=30),
            xaxis=dict(title="Tiempo (segundos)", gridcolor='#151515', zerolinecolor='#222'),
            yaxis=dict(title="Voltaje normalizado (mV)", gridcolor='#151515', zerolinecolor='#222')
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown(f"""
        <div style="background:#141414;border:1px solid #262626;border-radius:10px;padding:18px;margin:8px 0 22px 0;">
            <div style="font-size:13px;color:#ffffff;font-weight:700;margin-bottom:6px;">Qué demuestra este resultado</div>
            <div style="font-size:13px;color:#b8b8b8;line-height:1.6;">
                La señal original contiene ruido y variaciones que dificultan interpretar los latidos.
                Después del filtrado, los picos principales quedan más definidos y se pueden calcular
                indicadores como frecuencia cardíaca, regularidad y calidad de captura. Ese es el aporte
                central del proyecto: usar una herramienta matemática para convertir datos crudos en una lectura comprensible.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Explicación pedagógica
        col_exp1, col_exp2 = st.columns([1, 1])
        
        with col_exp1:
            st.markdown("""
            #### Filtro Notch: Elimina Ruido de Red Eléctrica
            
            **Problema**: Los electrodos captan interferencia de **60 Hz** (red eléctrica).
            
            **Solución**: El filtro Notch tiene un "agujero" (zero) en 60 Hz que bloquea ese ruido específicamente.
            
            $$H_{{Notch}}(s) = \\frac{s^2 + \\omega_0^2}{s^2 + \\frac{\\omega_0}{Q}s + \\omega_0^2}$$
            
            - El numerador crea un **cero** a 60 Hz (bloquea totalmente)
            - El denominador estabiliza la respuesta
            - Factor Q = 30 (filtro muy selectivo, solo afecta ±2 Hz alrededor de 60 Hz)
            """)
        
        with col_exp2:
            st.markdown("""
            #### Filtro Pasa-Banda: Mantiene Solo la Información Cardíaca
            
            **Rango clínico de ECG**: 0.5 – 40 Hz
            - **< 0.5 Hz**: Respiración, movimientos (lento)
            - **0.5 – 40 Hz**: Ondas P, QRS, T (cardíacas útiles)
            - **> 40 Hz**: Temblores musculares, artefactos (rápido)
            
            **Butterworth orden 4** (muy suave, sin oscilaciones):
            $$H_{{BP}}(s) = \\frac{s^2}{(s + p_1)(s + p_2)(s + p_3)(s + p_4)}$$
            
            Mantiene la forma natural del ECG sin distorsiones.
            """)
        
        st.divider()
        
        st.markdown("""
        #### Metrología de Calidad
        
        | Métrica | Valor | Interpretación |
        |---------|-------|----------------|
        | **SNR (Signal-to-Noise Ratio)** | {:.1f} dB | {} |
        | **Latidos detectados** | {} | Total en {} segundos |
        | **Anomalías encontradas** | {} | Latidos irregulares |
        """.format(
            f['snr_db'],
            "Excelente" if f['snr_db'] > 15 else ("Aceptable" if f['snr_db'] >= 5 else "Baja calidad"),
            f['total_beats'],
            f['duration_s'],
            f['anomalous_beats']
        ))


    # ==========================================
    # TAB 3: VISTA TÉCNICA / INGENIERÍA (ANÁLISIS DE LAPLACE)
    # ==========================================
    with tab3:
        st.markdown("### Herramientas avanzadas de ingeniería: análisis de Laplace")
        
        col_t1, col_t2 = st.columns([1, 1])
        
        with col_t1:
            st.markdown("#### Mapa de Polos y Ceros (Plano z)")
            st.markdown("""
            Después de aplicar la **transformación bilineal**, visualizamos los polos y ceros del filtro en el plano z.
            
            - **Ceros (o)**: Frecuencias que se **anulan** (60 Hz en el Notch)
            - **Polos (x)**: Frecuencias que se **amplifican**
            
            **Estabilidad**: Para que el filtro digital sea estable, todos los polos deben estar **dentro del círculo unitario**.
            """)
            # Dibujar mapa de polos y ceros
            fig_pz = plot_pole_zero_plotly(r['b_notch'], r['a_notch'], "Filtro Notch: Polos y Ceros (Dominio z)")
            st.plotly_chart(fig_pz, use_container_width=True)
            
        with col_t2:
            st.markdown("#### Respuesta en Frecuencia (Diagrama de Bode)")
            st.markdown("""
            Magnitud de los filtros (en decibelios dB). Muestra cuánto se atenúa o amplifica cada componente de frecuencia.
            
            - **Eje X**: Frecuencia (Hz)
            - **Eje Y**: Ganancia (dB)
            
            **Interpretación**:
            - **0 dB**: Sin cambio
            - **Negativo**: Atenuación (reducción)
            - **Banda de paso**: Mantiene la señal intacta
            - **Banda de rechazo**: Suprime completamente
            """)
            
            # Calcular respuesta de los filtros
            w_notch, h_notch = sp.freqz(r['b_notch'], r['a_notch'], worN=1024, fs=r['fs'])
            w_bp, h_bp = sp.freqz(r['b_bp'], r['a_bp'], worN=1024, fs=r['fs'])
            
            fig_bode = go.Figure()
            fig_bode.add_trace(go.Scatter(x=w_notch, y=20*np.log10(np.abs(h_notch)+1e-10), name='Filtro Notch', line=dict(color='#ffaa00', width=2)))
            fig_bode.add_trace(go.Scatter(x=w_bp, y=20*np.log10(np.abs(h_bp)+1e-10), name='Filtro Pasa-banda', line=dict(color='#00ccff', width=2)))
            
            # Marcar puntos críticos
            fig_bode.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="60 Hz", annotation_position="top right")
            fig_bode.add_vline(x=0.5, line_dash="dot", line_color="green", annotation_text="0.5 Hz", annotation_position="top right")
            fig_bode.add_vline(x=40, line_dash="dot", line_color="green", annotation_text="40 Hz", annotation_position="top right")
            
            fig_bode.update_layout(
                height=380,
                paper_bgcolor='#0a0a0a',
                plot_bgcolor='#0f0f0f',
                font=dict(color='#aaaaaa', size=10),
                legend=dict(bgcolor='rgba(17,17,17,0.8)', bordercolor='#222', borderwidth=1),
                margin=dict(l=40, r=20, t=30, b=30),
                xaxis=dict(title="Frecuencia (Hz)", range=[0, 100], gridcolor='#151515'),
                yaxis=dict(title="Magnitud (dB)", range=[-50, 5], gridcolor='#151515')
            )
            st.plotly_chart(fig_bode, use_container_width=True)
            
        st.markdown("---")
        
        st.markdown("### Fundamentos matemáticos detallados")
        
        col_math1, col_math2, col_math3 = st.columns(3)
        
        with col_math1:
            st.markdown("**1. Filtro Notch (Laplace)**")
            st.latex(r"H(s) = \frac{s^2 + \omega_0^2}{s^2 + \frac{\omega_0}{Q}s + \omega_0^2}")
            st.markdown("""
            - $\\omega_0 = 2\\pi f_0 = 2\\pi \\cdot 60 = 376.99$ rad/s
            - $Q = 30$ (factor de selectividad)
            - Ancho de banda: $BW = f_0/Q = 60/30 = 2$ Hz
            """)
        
        with col_math2:
            st.markdown("**2. Transformación Bilineal**")
            st.latex(r"s \mapsto \frac{2}{T} \cdot \frac{z-1}{z+1}")
            st.markdown(f"""
            donde $T = 1/f_s = 1/{r['fs']}$ segundos
            
            **Resultado**: H(s) → H(z) implementable digitalmente
            
            Ventajas:
            - Preserva estabilidad
            - Mapeo 1-a-1
            """)
        
        with col_math3:
            st.markdown("**3. Ecuación Diferencial de Diferencias**")
            st.latex(r"y[n] = b_0 x[n] + b_1 x[n-1] + ... - a_1 y[n-1] - ...")
            st.markdown(f"""
            Coeficientes Notch:
            - $b = {[f'{c:.6f}' for c in r['b_notch']]}$
            - $a = {[f'{c:.6f}' for c in r['a_notch']]}$
            """)
        
        st.markdown("---")
        
        st.markdown("### Datos extraídos para diagnóstico por IA")
        
        col_raw1, col_raw2 = st.columns([1, 1])
        
        with col_raw1:
            st.markdown("**Métricas Clínicas (Features)**")
            st.markdown(f"""
            | Parámetro | Valor | Rango Normal |
            |-----------|-------|-----|
            | FC promedio | {f['hr_mean']} bpm | 60–100 |
            | FC mín–máx | {f['hr_min']}–{f['hr_max']} bpm | Estable |
            | RMSSD | {f['rmssd']} ms | > 20 ms |
            | pNN50 | {f['pnn50']}% | > 10% |
            | QRS width | {f['qrs_width_ms']} ms | 80–120 |
            | SNR | {f['snr_db']} dB | > 10 |
            | Latidos totales | {f['total_beats']} | ~{ int(f['duration_s'] * f['hr_mean'] / 60)} |
            | Anomalías | {f['anomalous_beats']} | 0 |
            """)
        
        with col_raw2:
            st.markdown("**Prompt Completo → LLM (Ollama)**")
            from ecg_pipeline import build_medical_prompt
            prompt_text = build_medical_prompt(f)
            st.markdown("""
            <div style="background:#0f0f0f;border:1px solid #222;border-radius:8px;padding:12px;font-size:11px;font-family:monospace;color:#aaa;max-height:300px;overflow-y:auto;">
            """, unsafe_allow_html=True)
            st.code(prompt_text, language='text')
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### Notas para el trabajo académico")
        st.markdown(f"""
        - **Transformada de Laplace**: Aplicada a ambos filtros (Notch y Pasa-banda)  
        - **Transformación Bilineal**: Conversión s → z para implementación digital  
        - **MIT-BIH Database**: Registro {r['record']}  
        - **Análisis de Características**: RMSSD, pNN50, QRS width, SNR  
        - **Diagnóstico IA**: LLM local (Ollama) con {r['diagnosis'][:50] if r['diagnosis'] else 'pendiente'}...  
        
        **Referencias sugeridas para la memoria del trabajo:**
        - Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing*.
        - Pan, J., & Tompkins, W. J. (1985). A Real-Time QRS Detection Algorithm.
        - American Heart Association (2014). Recommended Standards for ECG Monitoring.
        """)


else:
    # Estado inicial de Bienvenida
    st.markdown("""
    <div style="
        background: radial-gradient(circle at top left, rgba(0, 204, 255, 0.07), transparent 60%),
                    radial-gradient(circle at bottom right, rgba(0, 255, 136, 0.03), transparent 60%),
                    #121212;
        border: 1px solid #252525;
        border-radius: 16px;
        padding: 40px 30px;
        text-align: center;
        max-width: 700px;
        margin: 40px auto;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    ">
        <h2 style="font-size: 26px; font-weight: 700; color: #ffffff; margin-bottom: 10px; letter-spacing: -0.5px;">
            Una lectura clara de una señal cardíaca
        </h2>
        <p style="font-size: 14px; color: #aaaaaa; max-width: 500px; margin: 0 auto 30px auto; line-height: 1.6;">
            Esta herramienta muestra cómo una señal con ruido puede convertirse en información útil:
            ritmo, regularidad, calidad de la lectura y una interpretación orientativa.
        </p>
        <div style="display: flex; flex-direction: column; gap: 12px; text-align: left; max-width: 480px; margin: 0 auto;">
            <div style="display: flex; align-items: center; gap: 16px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 10px;">
                <div style="background: linear-gradient(135deg, #00ccff, #0066ff); color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px;">1</div>
                <div style="font-size: 13px; color: #d0d0d0;">Usa la demostración o carga un registro para tener una señal de prueba.</div>
            </div>
            <div style="display: flex; align-items: center; gap: 16px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 10px;">
                <div style="background: linear-gradient(135deg, #00ff88, #00aa55); color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px;">2</div>
                <div style="font-size: 13px; color: #d0d0d0;">La app limpia el ruido y detecta los latidos principales.</div>
            </div>
            <div style="display: flex; align-items: center; gap: 16px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 10px;">
                <div style="background: linear-gradient(135deg, #ffaa00, #d48800); color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px;">3</div>
                <div style="font-size: 13px; color: #d0d0d0;">Lee primero el resumen fácil; la parte matemática queda como respaldo académico.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
