# Generador y Analizador de Senales Biomedicas ECG

Proyecto final de Transformada de Laplace para la asignatura de Ecuaciones Diferenciales / Cálculo IV.

La aplicacion permite generar, cargar, filtrar e interpretar senales ECG. El objetivo principal es mostrar como la Transformada de Laplace y las funciones de transferencia se pueden usar para modelar filtros que reducen ruido en una senal realista.

## Descripcion

Un electrocardiograma puede contener ruido electrico, movimiento, deriva de linea base o interferencias externas. Esto dificulta identificar los latidos y analizar el ritmo cardiaco.

Este proyecto toma una senal ECG cruda o simulada, aplica filtros digitales basados en el analisis de sistemas mediante Laplace, detecta picos R y genera un resumen facil de interpretar. Tambien permite descargar un reporte PDF con metricas y graficas principales.

El proyecto corresponde a la opcion de filtrado de senales, aplicada a una senal biomedica ECG.

## Objetivo general

Aplicar la Transformada de Laplace al filtrado de una senal biomedica ECG, mediante una simulacion computacional en Python que permita comparar la senal cruda con la senal filtrada e interpretar los resultados obtenidos.

## Objetivos especificos

- Representar el filtrado de senales mediante una ecuacion diferencial.
- Aplicar la Transformada de Laplace para obtener la funcion de transferencia del sistema.
- Simular o cargar una senal ECG en Python.
- Aplicar filtros para reducir ruido electrico y conservar la informacion cardiaca principal.
- Detectar picos R y calcular metricas basicas del ritmo cardiaco.
- Generar graficas y un reporte PDF con los resultados.

## Modelo matematico base

Un filtro de primer orden puede representarse mediante:

```text
tau dy/dt + y(t) = x(t)
```

Donde:

- `x(t)` es la senal de entrada.
- `y(t)` es la senal filtrada.
- `tau` es la constante de tiempo del sistema.

Aplicando Transformada de Laplace con condicion inicial cero:

```text
tau sY(s) + Y(s) = X(s)
```

Factorizando:

```text
Y(s)(tau s + 1) = X(s)
```

La funcion de transferencia queda:

```text
H(s) = Y(s) / X(s) = 1 / (tau s + 1)
```

Esta funcion permite analizar como la salida responde frente a la entrada y como el sistema suaviza cambios bruscos o componentes no deseadas de la senal.

## Aplicacion al ECG

En la aplicacion se utiliza el mismo principio de funcion de transferencia para filtrar una senal ECG:

- Filtro Notch: reduce interferencia de red electrica de 50 Hz o 60 Hz.
- Filtro pasa banda: conserva principalmente el rango util del ECG, entre 0.5 Hz y 40 Hz.
- Normalizacion: ajusta la escala de la senal filtrada.
- Deteccion de picos R: identifica los latidos principales.
- Analisis RR: permite observar la regularidad entre latidos.

## Funcionalidades

- Generacion de senal ECG sintetica con ruido controlado.
- Carga de registros MIT-BIH.
- Carga de archivos propios `.csv` o `.txt`.
- Filtrado Notch y pasa banda.
- Comparacion visual entre senal cruda y senal filtrada.
- Deteccion de picos R.
- Calculo de metricas:
  - Frecuencia cardiaca promedio.
  - Frecuencia minima y maxima.
  - RMSSD.
  - pNN50.
  - QRS estimado.
  - SNR.
  - Latidos irregulares.
- Grafica de intervalos RR.
- Interpretacion automatica opcional con Ollama.
- Descarga de reporte PDF.

## Estructura del proyecto

```text
Math/
├── app.py                  # Interfaz Streamlit
├── ecg_pipeline.py         # Procesamiento de senales ECG
├── INFORME_FINAL.md        # Informe escrito del proyecto
├── GUIA_RAPIDA.md          # Guia corta de uso
├── requirements.txt        # Dependencias completas
├── requirements_clean.txt  # Dependencias resumidas
├── data/                   # Registros ECG locales opcionales
└── .gitignore              # Archivos excluidos de Git
```

## Instalacion

Se recomienda crear un entorno virtual antes de instalar las dependencias.

```bash
python -m venv .venv
```

Activar el entorno en Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Tambien se puede usar el archivo resumido:

```bash
pip install -r requirements_clean.txt
```

## Ejecucion

```bash
streamlit run app.py
```

La aplicacion se abrira en el navegador, normalmente en:

```text
http://localhost:8501
```

## Uso recomendado para la sustentacion

1. Abrir la aplicacion con `streamlit run app.py`.
2. Usar la fuente "Demostracion" para asegurar una senal lista.
3. Ejecutar "Iniciar Analisis Completo".
4. Mostrar la pestana "Resumen facil".
5. Explicar la comparacion "Antes y despues".
6. Mostrar la grafica de intervalos RR.
7. Descargar el reporte PDF.
8. Ir a la vista tecnica para explicar la funcion de transferencia y la relacion con Laplace.

## Relacion con los entregables de la actividad

El proyecto incluye:

- Titulo del proyecto.
- Introduccion y descripcion de la situacion real.
- Ecuacion diferencial del sistema.
- Aplicacion de la Transformada de Laplace.
- Solucion matematica del modelo.
- Implementacion en Python.
- Graficas de resultados.
- Analisis e interpretacion.
- Conclusiones.
- Bibliografia.
- Anexos con codigo y evidencias sugeridas.

El informe principal se encuentra en:

```text
INFORME_FINAL.md
```

## Archivos que no se suben

El archivo `CAMBIOS_REALIZADOS.md` se usa solo como registro local de trabajo y no hace parte de la entrega final. Tambien se excluyen entornos virtuales, cache de Python y reportes generados.

## Bibliografia base

- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing. Prentice Hall.
- Pan, J., & Tompkins, W. J. (1985). A Real-Time QRS Detection Algorithm. IEEE Transactions on Biomedical Engineering.
- Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use.
- MIT-BIH Arrhythmia Database, PhysioNet.
- Documentacion oficial de SciPy Signal.
- Documentacion oficial de Streamlit.

## Nota

Esta aplicacion es educativa y orientativa. No reemplaza una valoracion medica profesional.

