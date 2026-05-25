# Filtrado y analisis de una senal biomedica ECG usando Transformada de Laplace

## 1. Introduccion al problema

Las senales biomedicas permiten estudiar el comportamiento del cuerpo humano mediante datos medibles. Una de las mas importantes es el electrocardiograma, conocido como ECG, que registra la actividad electrica del corazon.

En condiciones reales, una senal ECG puede verse afectada por ruido electrico, movimiento del paciente, mala ubicacion de electrodos o interferencias externas. Esto dificulta identificar correctamente los latidos y analizar parametros como la frecuencia cardiaca o la regularidad del ritmo.

Este proyecto aplica la Transformada de Laplace al filtrado de senales, con el fin de mostrar como una herramienta matematica permite modelar sistemas dinamicos, resolver ecuaciones diferenciales y mejorar una senal contaminada con ruido. La implementacion se realizo en Python mediante una aplicacion interactiva que genera o carga una senal ECG, aplica filtros digitales y presenta un resumen interpretable de los resultados.

## 2. Descripcion de la situacion real que se desea estudiar

La situacion estudiada corresponde al procesamiento de una senal cardiaca tipo ECG. En una medicion real, el registro no siempre aparece limpio. Puede contener componentes no deseadas, por ejemplo:

- Ruido de red electrica, comunmente de 50 Hz o 60 Hz.
- Deriva de linea base causada por respiracion o movimiento.
- Ruido muscular o pequenas variaciones de alta frecuencia.
- Cambios irregulares entre latidos.

El objetivo del sistema es tomar una senal ECG cruda, limpiarla mediante filtros y obtener una senal mas clara para detectar los picos R, que corresponden a los latidos principales. A partir de esos picos se calculan indicadores como frecuencia cardiaca, intervalos RR, variabilidad y calidad de la senal.

Este proyecto se relaciona con la opcion de filtrado de senales propuesta en la actividad, pero aplicada a una senal biomedica realista.

## 3. Ecuacion diferencial que representa el sistema

Un filtro pasa bajas de primer orden puede modelarse mediante la ecuacion diferencial:

```text
tau dy/dt + y(t) = x(t)
```

Donde:

- `x(t)` es la senal de entrada, en este caso la senal ECG cruda o contaminada.
- `y(t)` es la senal de salida, es decir, la senal filtrada.
- `tau` es la constante de tiempo del filtro.
- `dy/dt` representa la rapidez con la que cambia la salida.

Este modelo representa un sistema dinamico que no responde instantaneamente, sino que suaviza los cambios bruscos de la entrada. Por esta razon es util para reducir ruido en senales provenientes de sensores.

En el proyecto tambien se emplean filtros mas especificos para ECG:

- Filtro Notch para reducir la interferencia electrica de 50 Hz o 60 Hz.
- Filtro pasa banda para conservar principalmente la informacion cardiaca entre 0.5 Hz y 40 Hz.

La idea central es la misma: representar el sistema mediante una funcion de transferencia y analizar la relacion entre entrada y salida.

## 4. Aplicacion de la Transformada de Laplace

Partimos del modelo:

```text
tau dy/dt + y(t) = x(t)
```

Aplicando Transformada de Laplace en ambos lados:

```text
L{tau dy/dt + y(t)} = L{x(t)}
```

Usando la propiedad:

```text
L{dy/dt} = sY(s) - y(0)
```

Si se asume condicion inicial cero, es decir, `y(0) = 0`, se obtiene:

```text
tau sY(s) + Y(s) = X(s)
```

Factorizando `Y(s)`:

```text
Y(s)(tau s + 1) = X(s)
```

Por lo tanto, la funcion de transferencia del sistema es:

```text
H(s) = Y(s) / X(s) = 1 / (tau s + 1)
```

Esta expresion muestra como la Transformada de Laplace permite convertir una ecuacion diferencial en una relacion algebraica entre la entrada y la salida del sistema.

## 5. Solucion matematica del modelo

Si la entrada del sistema es un escalon de amplitud `A`, entonces:

```text
x(t) = A u(t)
```

Su Transformada de Laplace es:

```text
X(s) = A / s
```

Como:

```text
Y(s) = H(s)X(s)
```

Entonces:

```text
Y(s) = (1 / (tau s + 1)) (A / s)
```

```text
Y(s) = A / (s(tau s + 1))
```

Al aplicar fracciones parciales:

```text
Y(s) = A/s - A tau / (tau s + 1)
```

Aplicando Transformada Inversa de Laplace:

```text
y(t) = A(1 - e^(-t/tau))
```

Esta solucion indica que la salida del sistema se aproxima gradualmente al valor de entrada. La constante `tau` determina la rapidez de respuesta:

- Si `tau` es pequena, el sistema responde mas rapido.
- Si `tau` es grande, el sistema responde mas lento y suaviza mas la senal.

En el contexto del ECG, el filtrado permite atenuar componentes no deseadas y conservar la informacion util de la senal cardiaca.

## 6. Implementacion en Python

La implementacion se realizo en Python usando una aplicacion interactiva con Streamlit. El sistema permite tres formas de trabajo:

1. Generar una senal ECG sintetica con ruido controlado.
2. Cargar registros reales de la base MIT-BIH.
3. Subir un archivo propio en formato CSV o TXT.

El procesamiento principal se encuentra en el archivo `ecg_pipeline.py`, mientras que la interfaz se encuentra en `app.py`.

Flujo general del programa:

```text
Entrada ECG cruda
        |
        v
Filtro Notch 50/60 Hz
        |
        v
Filtro pasa banda 0.5 - 40 Hz
        |
        v
Normalizacion
        |
        v
Deteccion de picos R
        |
        v
Calculo de metricas
        |
        v
Reporte e interpretacion
```

Fragmento representativo del filtrado:

```python
def apply_filters(ecg_raw, fs=360, notch_freq=60.0, notch_q=30.0, lowcut=0.5, highcut=40.0):
    b_notch, a_notch = design_notch_filter(fs, notch_freq, notch_q)
    ecg_notched = signal.filtfilt(b_notch, a_notch, ecg_raw)

    b_bp, a_bp = design_bandpass_filter(fs, lowcut, highcut)
    ecg_filtered = signal.filtfilt(b_bp, a_bp, ecg_notched)

    ecg_normalized = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)

    return ecg_normalized, ecg_notched
```

Ademas, la aplicacion genera una grafica de intervalos RR y permite descargar un reporte PDF con los resultados principales.

## 7. Graficas de resultados

La aplicacion genera las siguientes graficas:

1. Senal ECG filtrada con picos R detectados.
2. Comparacion entre la senal cruda y la senal filtrada.
3. Grafica de intervalos RR para observar regularidad entre latidos.
4. Respuesta en frecuencia de los filtros.
5. Mapa de polos y ceros para analizar estabilidad.

Estas graficas permiten observar visualmente el efecto del filtrado. En la comparacion antes y despues, la senal cruda aparece con mayor ruido, mientras que la senal filtrada muestra los latidos de forma mas definida. La grafica de intervalos RR ayuda a interpretar si el ritmo es regular o si existen cambios importantes entre latidos.

## 8. Analisis e interpretacion

El filtrado de la senal ECG permite separar informacion util de componentes no deseadas. La Transformada de Laplace es importante porque permite representar el comportamiento del filtro mediante una funcion de transferencia, lo cual facilita estudiar como responde el sistema ante diferentes entradas.

En el caso del filtro de primer orden:

```text
H(s) = 1 / (tau s + 1)
```

se observa que el parametro `tau` controla la rapidez de respuesta. Un valor pequeno permite que la salida siga mas rapido a la entrada, mientras que un valor grande produce una respuesta mas suave.

En la aplicacion ECG, el filtro Notch elimina principalmente la interferencia electrica y el filtro pasa banda conserva el rango donde se encuentra la informacion cardiaca principal. Despues del filtrado, los picos R se identifican con mayor claridad, lo que permite calcular:

- Frecuencia cardiaca promedio.
- Intervalos RR.
- Variabilidad del ritmo.
- Latidos irregulares.
- Calidad aproximada de la senal.

La grafica de intervalos RR es especialmente util porque permite interpretar la regularidad del ritmo sin necesidad de conocer todos los detalles matematicos. Si la linea se mantiene relativamente estable, los latidos son mas regulares. Si aparecen saltos grandes, existe mayor variacion entre latidos.

## 9. Conclusiones

La Transformada de Laplace permite resolver ecuaciones diferenciales y representar sistemas dinamicos mediante funciones de transferencia. En este proyecto se aplico esta idea al filtrado de una senal biomedica ECG.

El modelo de filtro de primer orden muestra como una senal de entrada puede transformarse en una salida suavizada. A partir de este principio se implementaron filtros digitales para reducir ruido y mejorar la lectura de una senal cardiaca.

La simulacion en Python permitio visualizar la diferencia entre una senal cruda y una senal filtrada. Tambien permitio detectar latidos y calcular indicadores faciles de interpretar.

El proyecto demuestra que la Transformada de Laplace no es solo un procedimiento algebraico, sino una herramienta util para analizar sistemas reales, especialmente en procesamiento de senales, sensores y aplicaciones biomedicas.

## 10. Bibliografia consultada

- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing. Prentice Hall.
- Pan, J., & Tompkins, W. J. (1985). A Real-Time QRS Detection Algorithm. IEEE Transactions on Biomedical Engineering.
- Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use.
- MIT-BIH Arrhythmia Database, PhysioNet.
- Documentacion oficial de SciPy Signal.
- Documentacion oficial de Streamlit.

## 11. Anexos

### Anexo A. Archivos principales del proyecto

- `app.py`: interfaz de usuario, graficas, reporte PDF y controles.
- `ecg_pipeline.py`: generacion de senal, filtrado, deteccion de picos y calculo de metricas.
- `README.md`: documentacion general del proyecto.
- `GUIA_RAPIDA.md`: instrucciones de uso.
- `data/`: registros ECG usados para pruebas.

### Anexo B. Evidencias sugeridas para anexar

- Captura de la pantalla principal de la aplicacion.
- Captura de la senal ECG filtrada con picos detectados.
- Captura de la comparacion antes y despues del filtrado.
- Captura de la grafica de intervalos RR.
- PDF generado por la aplicacion.

### Anexo C. Comando de ejecucion

```bash
streamlit run app.py
```

