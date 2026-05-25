"""
Analizador Inteligente de Señales ECG
Transformada de Laplace aplicada al filtrado de señales cardíacas reales
MIT-BIH Arrhythmia Database + Ollama (LLM local)
"""

import numpy as np
import wfdb
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. DESCARGA DE DATOS MIT-BIH
# ─────────────────────────────────────────────

def download_record(record_name='100', save_dir='./data'):
    """Descarga un registro de PhysioNet MIT-BIH Arrhythmia Database."""
    os.makedirs(save_dir, exist_ok=True)
    try:
        files = [f"{record_name}.hea", f"{record_name}.dat", f"{record_name}.atr"]
        wfdb.dl_files('mitdb', save_dir, files)
        print(f"[OK] Registro {record_name} descargado en {save_dir}/")
    except Exception as e:
        print(f"[ERROR] No se pudo descargar: {e}")

def load_record(record_name='100', data_dir='./data'):
    """
    Carga la señal ECG y sus anotaciones desde MIT-BIH.
    Retorna: señal (numpy array), frecuencia de muestreo, anotaciones.
    """
    path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, 'atr')

    # Señal del canal 0 (MLII - derivación estándar)
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs  # 360 Hz para MIT-BIH
    return ecg_signal, fs, annotation


def generate_synthetic_ecg(duration_s=10, fs=360, hr=75, noise_powerline=0.2, noise_muscle=0.1, noise_drift=0.3, powerline_freq=60.0):
    """
    Genera un ECG sintético limpio con ruido controlable añadido:
    - hr: frecuencia cardíaca (bpm)
    - noise_powerline: nivel de ruido de línea (50/60 Hz)
    - noise_muscle: nivel de ruido muscular de alta frecuencia (EMG)
    - noise_drift: nivel de deriva de línea base (respiración)
    """
    t = np.arange(int(duration_s * fs)) / fs
    ecg_clean = np.zeros_like(t)

    # Período promedio en segundos
    rr_sec = 60.0 / hr
    
    # Posición de los latidos
    beat_times = []
    curr_time = 0.2  # Empezar un poco después de cero
    while curr_time < duration_s:
        beat_times.append(curr_time)
        # Añadir una pequeña variación fisiológica natural (variabilidad del ritmo)
        curr_time += rr_sec + np.random.normal(0, rr_sec * 0.04)

    # Parámetros de las ondas P, Q, R, S, T relativas al pico R (tiempo=0)
    # (Amplitud, delay en segundos, ancho en segundos)
    wave_params = {
        'P': (0.12, -0.15, 0.02),
        'Q': (-0.08, -0.05, 0.01),
        'R': (1.2, 0.0, 0.012),
        'S': (-0.25, 0.03, 0.015),
        'T': (0.3, 0.2, 0.035)
    }

    for bt in beat_times:
        for wave, (amp, delay, width) in wave_params.items():
            # Posición absoluta de la onda en la línea de tiempo
            center = bt + delay
            ecg_clean += amp * np.exp(-((t - center) ** 2) / (2 * (width ** 2)))

    # Agregar interferencia de línea eléctrica (Notch eliminará esto)
    interference = noise_powerline * np.sin(2 * np.pi * powerline_freq * t)

    # Agregar ruido muscular de alta frecuencia (pasa-banda eliminará esto)
    muscle_noise = noise_muscle * np.random.normal(0, 1, size=len(t))

    # Agregar deriva de línea de base (respiración) (pasa-banda / pasa-altas eliminará esto)
    # Representado por una onda senoidal lenta + tendencia lenta
    baseline_drift = noise_drift * np.sin(2 * np.pi * 0.15 * t) + noise_drift * 0.2 * np.sin(2 * np.pi * 0.03 * t)

    ecg_noisy = ecg_clean + interference + muscle_noise + baseline_drift

    return ecg_noisy, ecg_clean, t


def load_custom_file(file_obj):
    """
    Lee un archivo cargado por el usuario (CSV o TXT) que contiene una columna de datos.
    Intenta encontrar los datos numéricos.
    """
    import pandas as pd
    try:
        # Intentar leer con pandas
        # Si es un archivo de texto o csv, buscar números
        df = pd.read_csv(file_obj, header=None, comment='#', sep=None, engine='python')
        
        # Seleccionar la primera columna numérica
        for col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(numeric_col) > 10:
                return numeric_col.values
        raise ValueError("No se encontraron suficientes datos numéricos en el archivo.")
    except Exception as e:
        raise ValueError(f"Error parseando el archivo: {e}")



# ─────────────────────────────────────────────
# 2. FILTROS BASADOS EN TRANSFORMADA DE LAPLACE
#    H(s) → H(z) vía transformada bilineal
# ─────────────────────────────────────────────

def design_notch_filter(fs=360, f0=60.0, Q=30.0):
    """
    Filtro Notch para eliminar interferencia de red eléctrica (60 Hz).
    Aplicación de la Transformada de Laplace en filtrado digital.

    TEORÍA (Dominio s - Laplace):
    ═════════════════════════════════════
    H(s) = (s² + ω₀²) / (s² + (ω₀/Q)·s + ω₀²)
    
    donde:
        ω₀ = 2πf₀ (frecuencia angular a suprimir)
        Q = factor de calidad (mayor Q = mejor selectividad)
        BW = f₀/Q (ancho de banda de eliminación)
    
    IMPLEMENTACIÓN DIGITAL (Dominio z):
    ───────────────────────────────────
    Transformación Bilineal de Tustin:
        s → (2/T)·(z-1)/(z+1)
    
    donde T = 1/fs (período de muestreo)
    
    Parámetros:
        fs : frecuencia de muestreo (Hz)
        f0 : frecuencia a suprimir (60 Hz en América, 50 Hz en Europa)
        Q  : factor de calidad (30 es típico para ECG)

    Referencia: Oppenheim, A. V., & Schafer, R. W. (2010). 
    Discrete-Time Signal Processing. Prentice Hall.
    """
    w0 = f0 / (fs / 2)  # Frecuencia normalizada [0,1]
    b, a = signal.iirnotch(w0, Q)
    return b, a

def design_bandpass_filter(fs=360, lowcut=0.5, highcut=40.0, order=4):
    """
    Filtro pasa-banda Butterworth para ECG clínico.
    Filtra la banda de frecuencia donde reside la información cardíaca útil.

    TEORÍA (Dominio s - Laplace):
    ═════════════════════════════════════
    Butterworth pasa-banda orden n:
        H(s) = (Bw·sⁿ) / (s²ⁿ + √2·sⁿ⁺¹ + ...)
    
    donde:
        ω_L = 2π·lowcut (esquina inferior)
        ω_H = 2π·highcut (esquina superior)
        BW = ω_H - ω_L
        ω₀ = √(ω_L · ω_H) (frecuencia central geométrica)
    
    ESPECIFICACIÓN CLÍNICA ESTÁNDAR (AHA 2014):
    ────────────────────────────────────────────
    - Corte inferior: 0.5 Hz (remueve respiración, movimiento)
    - Corte superior: 40-100 Hz (remueve EMG, artefactos)
    - Para esta app: 0.5 Hz – 40 Hz, Butterworth orden 4
    
    JUSTIFICACIÓN:
    ──────────────
    Las ondas cardíacas P, QRS, T contienen energía en 0.5-40 Hz.
    Fuera de este rango solo hay ruido fisiológico o eléctrico.
    
    Parámetros:
        fs : frecuencia de muestreo (Hz)
        lowcut, highcut : frecuencias de corte (Hz)
        order : orden del filtro (4 = 2 cascadas de 2º orden)

    Referencia: American Heart Association (2014). 
    Recommended Standards for Electrocardiographic Monitoring.
    """
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_filters(ecg_raw, fs=360, notch_freq=60.0, notch_q=30.0, lowcut=0.5, highcut=40.0):
    """
    PIPELINE COMPLETO DE FILTRADO usando Transformada de Laplace.
    
    Arquitectura (en cascada):
    ══════════════════════════════════════════════════════════════════
    
    Entrada: x[n] (ECG crudo con ruido)
            ↓
        [FILTRO NOTCH (H₁)]
        Elimina 60/50 Hz (red eléctrica)
        H₁(s) = (s² + ω₀²) / (s² + (ω₀/Q)·s + ω₀²)
            ↓
        [FILTRO PASA-BANDA (H₂)]
        Mantiene 0.5-40 Hz (cardíaco)
        H₂(s) = Butterworth orden 4
            ↓
        [NORMALIZACIÓN Z-SCORE]
        Estandariza amplitud: (y - μ) / σ
            ↓
    Salida: y[n] (ECG limpio normalizado)
    
    TRANSFORMACIÓN DE LAPLACE A DIGITAL:
    ─────────────────────────────────────
    1. Diseñar H(s) en dominio continuo (Laplace)
    2. Aplicar transformación bilineal de Tustin
    3. Discretizar a H(z) para implementación digital
    4. Implementar con filtfilt (filtro de fase cero bidireccional)
    
    Parámetros: fs, notch_freq, notch_q, lowcut, highcut
    (ver docstrings de design_notch_filter y design_bandpass_filter)
    """
    # Paso 1: Eliminar interferencia eléctrica
    b_notch, a_notch = design_notch_filter(fs, notch_freq, notch_q)
    ecg_notched = signal.filtfilt(b_notch, a_notch, ecg_raw)

    # Paso 2: Eliminar deriva y artefactos musculares
    b_bp, a_bp = design_bandpass_filter(fs, lowcut, highcut)
    ecg_filtered = signal.filtfilt(b_bp, a_bp, ecg_notched)

    # Paso 3: Normalización (z-score)
    ecg_normalized = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)

    return ecg_normalized, ecg_notched


# ─────────────────────────────────────────────
# 3. EXTRACCIÓN DE FEATURES (CARACTERÍSTICAS)
# ─────────────────────────────────────────────

def detect_r_peaks(ecg_clean, fs=360):
    """
    Detecta picos R (complejos cardíacos) usando el algoritmo Pan-Tompkins.
    
    DEFINICIÓN FISIOLÓGICA:
    ══════════════════════
    El pico R es el punto de máxima amplitud positiva en cada complejo QRS.
    Representa la despolarización ventricular masiva (activación del ventrículo).
    
    ALGORITMO PAN-TOMPKINS (1985):
    ───────────────────────────────
    1. Búsqueda de máximos locales en la señal filtrada
    2. Distancia mínima de 0.25s entre picos (evita doble detección, max ~240 bpm)
    3. Umbral adaptativo: 60% del percentil 95 de amplitud
    
    RESULTADOS:
    ───────────
    Retorna índices de muestras (array numpy) donde están los picos R.
    Estos índices son cruciales para calcular:
        - Intervalos RR (tiempo entre latidos)
        - Variabilidad de ritmo cardíaco (HRV)
        - Frecuencia cardíaca (HR)
    
    Referencia: Pan, J., & Tompkins, W. J. (1985). 
    A Real-Time QRS Detection Algorithm. IEEE Trans Biomed Eng.
    """
    # Distancia mínima entre picos: 0.25 segundos (evita doble detección, soporta hasta 240 bpm)
    min_distance = int(0.25 * fs)

    # Umbral adaptativo: 60% del percentil 95
    threshold = 0.6 * np.percentile(ecg_clean, 95)

    peaks, properties = signal.find_peaks(
        ecg_clean,
        height=threshold,
        distance=min_distance
    )
    return peaks

def calculate_features(ecg_clean, peaks, fs=360, duration_s=10):
    """
    Calcula las MÉTRICAS CLÍNICAS estándar (HRV de Tiempo y Dominio de Frecuencia).
    
    MÉTRICAS DE DOMINIO TEMPORAL (time-domain):
    ════════════════════════════════════════════
    
    1. HR (Frecuencia Cardíaca):
       FC = 60 / (RR_promedio / 1000) [bpm]
       Rango normal: 60-100 bpm en reposo
    
    2. RMSSD (Root Mean Square of Successive Differences):
       RMSSD = √(Σ(RRᵢ - RRᵢ₊₁)² / n)
       Indicador de: actividad parasimpática (vagal)
       > 40 ms → excelente adaptabilidad autonómica
    
    3. pNN50 (Percentage of NN intervals > 50ms):
       pNN50 = (# {RRᵢ - RRᵢ₊₁ > 50ms} / n) × 100
       Correlaciona con adaptabilidad al estrés
    
    4. QRS Width (Duración del complejo QRS):
       Medida del ancho de despolarización ventricular
       Normal: 80-120 ms | > 120 ms → potencial bloqueo
    
    5. SNR (Signal-to-Noise Ratio):
       SNR_dB = 10·log₁₀(P_señal / P_ruido)
       Indicador de calidad de grabación
    
    6. Detección de Latidos Anómalos:
       Latidos cuyo RR se desvía > 2σ del promedio (outliers)
    
    REFERENCIAS CLÍNICAS:
    ────────────────────
    - Heart Rate Variability: Standards of Measurement, 
      Physiological Interpretation, and Clinical Use. 
      Task Force of the European Society of Cardiology (1996)
    - Medtronic HRV Guidelines for Clinical Use (2019)
    
    Retorna: diccionario con todas las features para diagnóstico IA
    """
    if len(peaks) < 2:
        return None

    # Intervalos RR en segundos
    rr_intervals = np.diff(peaks) / fs * 1000  # en milisegundos

    # Frecuencia cardíaca media
    hr_mean = 60 / (np.mean(rr_intervals) / 1000)

    # RMSSD: raíz cuadrada de la media de diferencias cuadradas de RR consecutivos
    # Indicador de variabilidad del sistema nervioso autónomo
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))

    # pNN50: porcentaje de intervalos RR consecutivos que difieren más de 50 ms
    pnn50 = (np.sum(np.abs(rr_diff) > 50) / len(rr_diff)) * 100

    # Detección de latidos anómalos (intervalo RR muy diferente al promedio)
    rr_mean = np.mean(rr_intervals)
    rr_std = np.std(rr_intervals)
    anomalous_beats = np.sum(np.abs(rr_intervals - rr_mean) > 2 * rr_std)

    # Ancho aproximado del complejo QRS (en ms)
    qrs_width = estimate_qrs_width(ecg_clean, peaks, fs)

    # SNR (relación señal/ruido en dB)
    signal_power = np.mean(ecg_clean ** 2)
    noise_est = np.std(np.diff(ecg_clean)) / np.sqrt(2)
    snr_db = 10 * np.log10(signal_power / (noise_est ** 2 + 1e-10))

    return {
        'hr_mean': round(hr_mean, 1),
        'hr_min': round(60 / (np.max(rr_intervals) / 1000), 1),
        'hr_max': round(60 / (np.min(rr_intervals) / 1000), 1),
        'rmssd': round(rmssd, 2),
        'pnn50': round(pnn50, 1),
        'qrs_width_ms': round(qrs_width, 1),
        'anomalous_beats': int(anomalous_beats),
        'total_beats': len(peaks),
        'snr_db': round(snr_db, 1),
        'rr_mean_ms': round(rr_mean, 1),
        'rr_std_ms': round(rr_std, 1),
        'duration_s': duration_s
    }

def estimate_qrs_width(ecg_clean, peaks, fs):
    """
    ESTIMACIÓN DEL ANCHO DEL COMPLEJO QRS.
    
    DEFINICIÓN CLÍNICA:
    ─────────────────────────────────────────────────────────────────────
    El complejo QRS representa la despolarización ventricular rápida.
    Su duración se mide entre el inicio de Q y el final de S.
    
    - **Normal**: 80–120 ms
    - **Anormal (> 120 ms)**: Sugiere bloqueo intraventricular,
      hipertrofia ventricular, o arritmia.
    
    ALGORITMO DE ESTIMACIÓN:
    ─────────────────────────────────────────────────────────────────────
    Para cada pico R detectado:
    1. Definir ventana de ±50 ms alrededor del pico
    2. Encontrar donde la amplitud supera el 30% del pico
    3. Medir el tiempo entre esos puntos (ancho)
    4. Promediar sobre los primeros 20 latidos
    
    Parámetros:
        ecg_clean : señal filtrada normalizada
        peaks : índices de picos R
        fs : frecuencia de muestreo (Hz)
    
    Retorna:
        float : duración promedio del QRS en milisegundos
    """
    widths = []
    window = int(0.05 * fs)  # 50ms ventana
    threshold = 0.3

    for peak in peaks[:20]:  # Muestra de los primeros 20 latidos
        start = max(0, peak - window)
        end = min(len(ecg_clean), peak + window)
        segment = ecg_clean[start:end]
        peak_val = ecg_clean[peak]

        # Ancho donde la señal supera el 30% del pico
        above = np.where(segment > threshold * peak_val)[0]
        if len(above) > 0:
            widths.append((above[-1] - above[0]) / fs * 1000)

    return np.mean(widths) if widths else 80.0


# ─────────────────────────────────────────────
# 4. DIAGNÓSTICO CON OLLAMA (LLM LOCAL)
# ─────────────────────────────────────────────

OLLAMA_SYSTEM_PROMPT = """Eres un sistema de apoyo diagnóstico cardiológico basado en análisis de señales ECG.
Tu rol es interpretar parámetros cuantitativos extraídos de registros electrocardiográficos
y generar un informe diagnóstico conciso en español.

Reglas:
- Responde SOLO con el diagnóstico, sin introducción ni despedida
- Usa terminología clínica apropiada
- Menciona si el patrón es normal o anormal
- Si detectas anomalías, nómbralas con su término médico correcto
- Máximo 5 oraciones
- NO inventes valores, trabaja solo con los datos proporcionados"""

def build_medical_prompt(features):
    """
    Construye el prompt médico con las features extraídas.
    Este es el puente entre el procesamiento de señales y el LLM.
    """
    return f"""Analiza los siguientes parámetros de un ECG de {features['duration_s']} segundos:

PARÁMETROS TEMPORALES:
- Frecuencia cardíaca media: {features['hr_mean']} bpm (rango: {features['hr_min']}–{features['hr_max']} bpm)
- Intervalo RR medio: {features['rr_mean_ms']} ms (desviación estándar: {features['rr_std_ms']} ms)

VARIABILIDAD (HRV):
- RMSSD: {features['rmssd']} ms
- pNN50: {features['pnn50']}%

MORFOLOGÍA:
- Duración compleja QRS: {features['qrs_width_ms']} ms
- Latidos totales detectados: {features['total_beats']}
- Latidos con intervalo RR anómalo (>2σ): {features['anomalous_beats']}

CALIDAD DE SEÑAL:
- SNR: {features['snr_db']} dB

Emite el diagnóstico:"""

def diagnose_with_ollama(features, model='llama3.1:8b', host='http://127.0.0.1:11434'):
    """
    Envía las features al LLM local (Ollama) y retorna el diagnóstico.

    Ollama corre completamente local — no envía datos a ningún servidor externo.
    Compatible con GPU NVIDIA mediante CUDA.
    """
    prompt = build_medical_prompt(features)

    payload = {
        "model": model,
        "system": OLLAMA_SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,   # Baja temperatura = respuestas más deterministas
            "num_predict": 256    # Máximo de tokens en la respuesta
        }
    }

    try:
        response = requests.post(
            f"{host}/api/generate",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get('response', 'Sin respuesta del modelo.')
        else:
            try:
                err_detail = response.json().get('error', response.text)
            except Exception:
                err_detail = response.text
            return f"[Error HTTP {response.status_code}] Ollama reporta: {err_detail}"
    except requests.exceptions.ConnectionError:
        return "[Error de conexión] Ollama no está corriendo. Ejecuta: ollama serve"
    except Exception as e:
        return f"[Error] {str(e)}"


# ─────────────────────────────────────────────
# 5. VISUALIZACIÓN
# ─────────────────────────────────────────────

def plot_ecg_analysis(ecg_raw, ecg_clean, peaks, features, anomalous_peaks,
                      fs=360, duration_s=10, save_path='./outputs/ecg_analysis.png'):
    """
    Genera figura de análisis completo:
      - Panel superior: señal cruda vs filtrada
      - Panel central: señal limpia con picos R y zonas anómalas
      - Panel inferior: respuesta en frecuencia del filtro Notch (H(s) → H(z))
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs

    ecg_raw_seg = ecg_raw[:n_samples]
    ecg_clean_seg = ecg_clean[:n_samples]
    peaks_seg = peaks[peaks < n_samples]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f0f')
    for ax in axes:
        ax.set_facecolor('#0f0f0f')
        ax.tick_params(colors='#aaaaaa')
        ax.spines['bottom'].set_color('#333333')
        ax.spines['top'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        ax.spines['right'].set_color('#333333')

    # Panel 1: Comparación señal cruda vs filtrada
    ax1 = axes[0]
    ax1.plot(t, ecg_raw_seg, color='#ff4444', alpha=0.6, linewidth=0.8, label='Señal cruda (con ruido)')
    ax1.plot(t, ecg_clean_seg, color='#00ff88', linewidth=1.0, label='Señal filtrada')
    ax1.set_title('Comparación: antes y después del filtrado (Laplace → bilineal)',
                  color='white', fontsize=11, pad=8)
    ax1.set_ylabel('Amplitud (mV)', color='#aaaaaa', fontsize=9)
    ax1.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='#333333',
               labelcolor='white', fontsize=8)
    ax1.grid(True, color='#1a1a1a', linewidth=0.5)

    # Panel 2: ECG limpio con detección de picos R y anomalías
    ax2 = axes[1]
    ax2.plot(t, ecg_clean_seg, color='#00ccff', linewidth=1.0, label='ECG filtrado')

    # Marcar todos los picos R en verde
    ax2.scatter(peaks_seg / fs, ecg_clean_seg[peaks_seg],
                color='#00ff44', s=40, zorder=5, label='Picos R detectados')

    # Marcar latidos anómalos en rojo
    anom_in_seg = [p for p in anomalous_peaks if p < n_samples]
    if anom_in_seg:
        ax2.scatter(np.array(anom_in_seg) / fs,
                    ecg_clean_seg[anom_in_seg],
                    color='#ff3333', s=80, marker='x', linewidths=2,
                    zorder=6, label=f'Latidos anómalos ({len(anom_in_seg)})')
        # Zona de alerta
        for ap in anom_in_seg:
            ax2.axvspan(ap/fs - 0.15, ap/fs + 0.15,
                        alpha=0.15, color='red', zorder=1)

    ax2.set_title(f'Detección de picos R | FC: {features["hr_mean"]} bpm | '
                  f'Anomalías: {features["anomalous_beats"]} latidos',
                  color='white', fontsize=11, pad=8)
    ax2.set_ylabel('Amplitud (normalizada)', color='#aaaaaa', fontsize=9)
    ax2.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='#333333',
               labelcolor='white', fontsize=8)
    ax2.grid(True, color='#1a1a1a', linewidth=0.5)

    # Panel 3: Respuesta en frecuencia del filtro Notch H(jω)
    ax3 = axes[2]
    b_notch, a_notch = design_notch_filter(fs)
    w, h = signal.freqz(b_notch, a_notch, worN=2048, fs=fs)
    ax3.plot(w, 20 * np.log10(np.abs(h) + 1e-10),
             color='#ffaa00', linewidth=1.2, label='|H(jω)| Filtro Notch 60 Hz')
    ax3.axvline(x=60, color='#ff4444', linestyle='--', alpha=0.7, label='60 Hz (suprimida)')
    ax3.axvline(x=0.5, color='#44ff88', linestyle=':', alpha=0.5, label='0.5 Hz (pasa-banda inf.)')
    ax3.axvline(x=40, color='#44ff88', linestyle=':', alpha=0.5, label='40 Hz (pasa-banda sup.)')
    ax3.set_xlim([0, 100])
    ax3.set_ylim([-50, 5])
    ax3.set_title('Respuesta en frecuencia del filtro — derivada de H(s) = (s²+ω₀²)/(s²+Bw·s+ω₀²)',
                  color='white', fontsize=11, pad=8)
    ax3.set_xlabel('Frecuencia (Hz)', color='#aaaaaa', fontsize=9)
    ax3.set_ylabel('Magnitud (dB)', color='#aaaaaa', fontsize=9)
    ax3.legend(loc='lower right', facecolor='#1a1a1a', edgecolor='#333333',
               labelcolor='white', fontsize=8)
    ax3.grid(True, color='#1a1a1a', linewidth=0.5)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
    plt.close()
    print(f"[OK] Figura guardada en {save_path}")
    return save_path


# ─────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_full_pipeline(record_name='100', data_dir='./data',
                      duration_s=10, use_ollama=True):
    """
    Ejecuta el pipeline completo:
      1. Carga señal MIT-BIH
      2. Aplica filtros Laplace
      3. Detecta picos R y extrae features
      4. Diagnóstico con Ollama
      5. Genera visualización
    """
    print(f"\n{'='*55}")
    print(f"  ANALIZADOR ECG — Registro MIT-BIH {record_name}")
    print(f"{'='*55}\n")

    # Paso 1: Cargar datos
    print("[1/5] Cargando registro ECG...")
    ecg_raw, fs, annotation = load_record(record_name, data_dir)
    n_samples = int(duration_s * fs)
    ecg_segment = ecg_raw[:n_samples]
    print(f"      Fs={fs} Hz | {duration_s}s | {n_samples} muestras")

    # Paso 2: Filtrar
    print("[2/5] Aplicando filtros Laplace (Notch 60Hz + pasa-banda)...")
    ecg_clean, _ = apply_filters(ecg_segment, fs)
    print(f"      SNR mejorado: señal procesada con filtro bilineal")

    # Paso 3: Detectar picos y calcular features
    print("[3/5] Detectando picos R y calculando features...")
    peaks = detect_r_peaks(ecg_clean, fs)
    features = calculate_features(ecg_clean, peaks, fs, duration_s)

    if features is None:
        print("[ERROR] No se detectaron suficientes picos R.")
        return

    # Identificar latidos anómalos
    rr = np.diff(peaks)
    rr_mean, rr_std = np.mean(rr), np.std(rr)
    anom_idx = np.where(np.abs(rr - rr_mean) > 2 * rr_std)[0]
    anomalous_peaks = peaks[anom_idx + 1].tolist()

    print(f"      FC: {features['hr_mean']} bpm | Latidos: {features['total_beats']} | "
          f"Anomalías: {features['anomalous_beats']}")

    # Paso 4: Diagnóstico IA
    print("[4/5] Enviando features a Ollama (LLM local)...")
    if use_ollama:
        diagnosis = diagnose_with_ollama(features)
    else:
        diagnosis = "[Modo offline] Ollama no habilitado. Activa use_ollama=True"
    print(f"\nDIAGNÓSTICO:\n{'-'*40}\n{diagnosis}\n{'-'*40}\n")

    # Paso 5: Visualización
    print("[5/5] Generando gráficas...")
    fig_path = plot_ecg_analysis(
        ecg_segment, ecg_clean, peaks, features, anomalous_peaks,
        fs=fs, duration_s=duration_s
    )

    return {
        'features': features,
        'diagnosis': diagnosis,
        'figure_path': fig_path,
        'peaks': peaks,
        'anomalous_peaks': anomalous_peaks
    }


# ─────────────────────────────────────────────
# ENTRADA PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Descarga el registro si no existe
    if not os.path.exists('./data/100.dat'):
        print("Descargando registro 100 de MIT-BIH PhysioNet...")
        download_record('100', './data')

    # Ejecutar análisis completo
    # Para la demo puedes cambiar el record a '101', '200', '201'
    results = run_full_pipeline(
        record_name='100',
        data_dir='./data',
        duration_s=10,
        use_ollama=True  # Cambia a False si Ollama no está corriendo
    )
