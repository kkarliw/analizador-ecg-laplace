# Guía Rápida de Uso

## Instalación (1 minuto)

```bash
# 1. Abrir terminal en la carpeta Math
cd Math

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (OPCIONAL) Para diagnóstico IA - Ollama
# Descargar desde https://ollama.ai/
# Luego ejecutar en OTRA terminal:
# ollama serve
```

## Ejecución (30 segundos)

```bash
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`

---

## Cómo Usar la App

### Paso 1: Selecciona Fuente de Datos (Panel Izquierdo)

Tres opciones:

- **Demostración** (recomendado para empezar)
  - ECG sintético generado automáticamente
  - Listo para usar sin descargas

- **MIT-BIH Database**
  - Registros cardíacos reales
  - Se descargan automáticamente (~1 MB por registro)
  - Opciones: Ritmo normal, Latidos irregulares, Taquicardia, etc.

- **Mi Propio Archivo** 
  - CSV o TXT con una columna numérica
  - Cada línea = una muestra de voltaje
  - Especifica velocidad de captura (Hz)

### ⚙️ Paso 2: Ajusta Parámetros (Opcional)

**Configuración Básica:**
- `Segundos a analizar`: 5–60 segundos de ECG
- Dejar el resto por defecto es suficiente

**Configuración Avanzada** (expandir si quieres tuning):
- `Simulador`: HR, ruido eléctrico, muscular, respiración
- `Filtros`: Frecuencias notch y pasa-banda
- `IA`: Modelo de Ollama, servidor

### ▶️ Paso 3: Ejecuta Análisis

Haz clic en el botón azul grande:
> **"Iniciar Análisis Completo"**

Espera 3–30 segundos según la fuente:
- Simulador: ~3 segundos
- MIT-BIH: ~10 segundos (primera vez, descarga datos)
- Ollama: +10–20 segundos (genera diagnóstico)

### 📊 Paso 4: Visualiza Resultados (3 Pestañas)

#### 🏥 Pestaña 1: **Reporte de Salud**
- **Para No-Técnicos**: Indicadores en lenguaje simple
  - ✓ Ritmo cardíaco (bpm)
  - ✓ Regularidad del latido
  - ✓ Adaptabilidad cardíaca (HRV)
  - ✓ Calidad de grabación

- Gráfico ECG limpio con latidos marcados
- Interpretación IA (si está disponible)

#### 🔬 Pestaña 2: **Señal y Filtrado**
- **Para Estudiantes de Ingeniería**: Cómo funcionan los filtros
  - Comparación antes/después
  - Explicación de Filtro Notch (elimina 60 Hz)
  - Explicación de Filtro Pasa-Banda (mantiene cardíaco)
  - Tabla de métricas

#### 🔧 Pestaña 3: **Vista Técnica**
- **Para Trabajo Académico**: Análisis profundo
  - Mapa de Polos y Ceros (plano z)
  - Diagrama de Bode (respuesta en frecuencia)
  - Fórmulas matemáticas (LaTeX)
  - Coeficientes de filtros (b, a)
  - Features completas (todo para la IA)
  - Prompt enviado al LLM

---

## 📝 Ejemplo Paso a Paso

### Ejemplo 1: Análisis Rápido (Demostración)

```
1. Panel izquierdo:
   ✓ "Demostración (generada automáticamente)" (ya está seleccionada)
   ✓ Dejar "Segundos a analizar" en 10

2. Haz clic: "Iniciar Análisis Completo"

3. En 3 segundos ves:
   ✓ Pestaña "Reporte de Salud" se abre automáticamente
   ✓ 4 tarjetas coloridas con tu ritmo cardíaco simulado
   ✓ Gráfico ECG limpio
   ✓ Diagnóstico IA (si Ollama está corriendo)

4. Explora otras pestañas para ver filtros y fórmulas
```

### Ejemplo 2: Análisis de Registro Real (MIT-BIH)

```
1. Panel izquierdo:
   → Cambiar a "Base de datos clínica (MIT-BIH)"
   → Seleccionar: "Ritmo normal" (registro 100)
   → Dejar 10 segundos

2. Haz clic: "Iniciar Análisis Completo"

3. Primera ejecución:
   ✓ Descarga registro 100 (~1 MB) — espera 5–10 seg
   ✓ Procesa señal real
   ✓ Muestra resultados

4. Próximas ejecuciones del mismo registro:
   ✓ Instant (datos ya descargados en ./data/)
```

### Ejemplo 3: Análisis de Tu Archivo

```
1. Panel izquierdo:
   → Cambiar a "Mi propio archivo"
   → Haz clic en "Selecciona tu archivo"
   → Sube archivo.csv o archivo.txt

2. Formato del archivo:
   ```
   120.5
   119.8
   121.2
   ...
   ```
   (un número por línea = voltaje)

3. Si no sabes la velocidad de captura:
   → Dejar en 250 Hz (valor por defecto)

4. Haz clic: "Iniciar Análisis Completo"
   ✓ Procesa tu archivo
   ✓ Muestra resultados
```

---

## ⚠️ Soluciones a Problemas

### ❌ Error: "ModuleNotFoundError: No module named 'streamlit'"

**Solución:**
```bash
pip install -r requirements.txt
```

### ❌ Error: "[Error de conexión] Ollama no está corriendo"

**Solución:**
- Abre OTRA terminal
- Ejecuta: `ollama serve`
- O desactiva IA: Configuración avanzada → Desactiva "Activar diagnóstico IA"

### ❌ Error: "No se encontraron suficientes datos numéricos en el archivo"

**Solución:**
- Tu archivo CSV/TXT no tiene el formato correcto
- Verifica que cada línea tenga un número
- Prueba primero con Demostración

### 🐢 "Está muy lento"

**Posibles causas:**
- Ollama procesando (puede ser lento en CPU)
- Primera ejecución MIT-BIH (descargando datos)
- Archivo muy grande

**Soluciones:**
- Reducir "Segundos a analizar" a 5–10 segundos
- Desactivar Ollama (ir a Configuración Avanzada)
- Usar GPU si está disponible (Ollama + NVIDIA CUDA)

---

## 🎯 Consejos para Trabajo Académico

### Para Explicar Laplace

**Usa Pestaña "Señal y Filtrado":**
- Muestra visualmente cómo Laplace limpia ruido
- Explica qué es un filtro Notch
- Explica qué es pasa-banda

### Para Mostrar Matemática

**Usa Pestaña "Vista Técnica":**
- Captura pantalla de fórmulas (LaTeX)
- Muestra Diagrama de Bode
- Explica polos y ceros
- Copia coeficientes b, a para el informe

### Para Demostración en Clase

1. Ejecuta app en modo demostración
2. Usa registro MIT-BIH 101 (arritmia interesante)
3. Muestra 3 pestañas en orden
4. Pregunta: "¿Ven cómo el ruido rojo desaparece en verde?" (Pestaña 2)

### Para el Informe Final

**Captura de pantalla útiles:**
- Tab 1 completa (indicadores + ECG)
- Tab 2 con gráficos comparativos
- Tab 3 con fórmulas y polos-ceros

**Datos para copiar:**
- Tabla de features (tab 3, lado izquierdo)
- Prompt a IA (tab 3, lado derecho)
- Coeficientes de filtros

---

## 📱 Interfaz Visual Rápida

```
┌─────────────────────────────────────────────────────────────────┐
│  🏥 Analizador Inteligente de Señales ECG                       │
│  📚 Transformada de Laplace en Filtrado Cardíaco               │
│                                              [FUENTE: SIMULADOR] │
│                                                     [360 Hz · Fs] │
└──────────────────────────────────────────────────────────────────┘

┌─ PANEL IZQUIERDO ──┐  ┌─ PANEL PRINCIPAL ──────────────────────┐
│                    │  │                                         │
│ Analizador ECG     │  │ 📚 ¿Cómo funciona?                    │
│ ───────────────    │  │    [expandir para ver Laplace]       │
│ Fuente:            │  │                                         │
│ ◉ Demostración     │  │ [Introduc. a Laplace]                 │
│ ○ MIT-BIH          │  │                                         │
│ ○ Mi archivo       │  │ ───────────────────────────────────── │
│                    │  │                                         │
│ Segundos: [10]     │  │ 🟦 Iniciar Análisis Completo          │
│                    │  │                                         │
│ ⚙️ Avanzado        │  │ ───────────────────────────────────── │
│    [expandir]      │  │                                         │
│                    │  │ 📊 [Reporte] [Señal] [Técnica]         │
│                    │  │                                         │
│                    │  │ [Contenido de pestaña seleccionada]    │
│                    │  │                                         │
│                    │  │                                         │
└────────────────────┘  └─────────────────────────────────────────┘
```

---

## 📚 Más Información

- **README.md**: Marco teórico completo, referencias académicas
- **app.py**: Código fuente del frontend
- **ecg_pipeline.py**: Código del procesamiento de señales
- **requirements.txt**: Dependencias

---

**¡Lista para usar!** 🚀  
*Si algo no funciona, revisa "Soluciones a Problemas" arriba.*
