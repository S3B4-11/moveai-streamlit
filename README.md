# MOVEAI en Streamlit

Este paquete usa los cuatro modelos del ZIP MOVEAI_privado_actualizado que exportaste en Colab. No vuelve a entrenar ni descarga los datasets al abrir la app. Incluye el detector de pose para evitar una descarga al arrancar. La calibración del semáforo ya está calculada a partir de tus predicciones de validación guardadas en Drive.

## GitHub y Streamlit Cloud

1. Descomprime este ZIP. Crea un repositorio nuevo en GitHub, por ejemplo moveai-streamlit.
2. Sube el CONTENIDO de la carpeta a la raíz del repositorio: streamlit_app.py, los módulos .py, requirements.txt, packages.txt, assets/, modelos_rapidos/ y .streamlit/config.toml. Sube las carpetas completas. No subas el ZIP como único archivo, videos, revisiones ni contraseñas.
3. En https://share.streamlit.io elige Create app → Yup, I have an app.
4. Selecciona tu repositorio, la rama main y el archivo streamlit_app.py.
5. En Advanced settings selecciona Python 3.12 y añade en Secrets:

```toml
app_password = "TU_CONTRASENA_DE_AL_MENOS_8_CARACTERES"
```

6. Guarda y despliega la app. Entra usando esa contraseña y comparte el enlace con el equipo.

Si el archivo streamlit_app.py quedó dentro de una carpeta en GitHub, usa esa ruta completa como entrada. Es más sencillo subir el contenido directamente a la raíz. En un repositorio existente, sustituye también los módulos y requirements: no mezcles las dependencias antiguas de TensorFlow con estas.

## Prueba inicial

Graba una repetición completa con buena luz y cámara fija. Para hombro elige frente; para medir recorrido de rodilla o codo, perfil. Selecciona el lado activo y el objetivo del ejercicio. Sube el video, pulsa Analizar y revisa medidas, criterios sin evaluar y sugerencias. Compara una repetición completa y una de recorrido claramente reducido bajo supervisión; MOVEAI debería detectar esa diferencia cuando las articulaciones sean visibles. Una diferencia entre la app y el revisor se conserva como error para evaluar.

La app muestra VERDE y «Técnica clasificada como correcta» si la probabilidad estimada de correcta alcanza 75%, hay movimiento, articulaciones visibles, características dentro del rango de entrenamiento y ninguna desviación o medida cercana al límite. ROJO aparece con una desviación observada o con probabilidad estimada de incorrecta de al menos 75%. AMARILLO indica que falta calidad, evidencia o confianza suficiente. Un criterio 2D no disponible por la vista no oculta la clasificación 3D si los datos del modelo son suficientes. Si falta una calibración válida, la señal anterior puede aparecer en amarillo, sin un porcentaje.

La probabilidad usa una transformación sigmoide ajustada a los scores OOF, con prevalencia de los datasets de referencia. Los cortes del semáforo reemplazan los cortes anteriores de detección para la decisión coloreada; las métricas binarias de modelo.json siguen siendo las del entrenamiento anterior. No se modificaron los árboles ni se reentrenaron los modelos. La calibración no mejora necesariamente Brier o AUC: su ajuste se audita, no se asume.

Comprobación posterior del score, ajustando el calibrador sin usar etiquetas del fold de personas evaluado:

| Ejercicio | Verdes correctos / verdes | Acierto entre verdes | Amarillos / total |
|---|---:|---:|---:|
| Sentadilla | 126 / 144 | 87,5% | 165 / 371 |
| Zancada | 63 / 84 | 75,0% | 90 / 275 |
| Hombro | 476 / 509 | 93,5% | 184 / 811 |
| Codo | 413 / 437 | 94,5% | 36 / 520 |

Estos números son precisión de los verdes del score, excluyen amarillos y no incluyen los controles de visibilidad, dominio o criterios 2D aplicados por la app. No son accuracy global ni una nueva prueba independiente: los modelos base que generaron scores de otros folds pueden haber visto etiquetas del fold evaluado. No es una CV completamente anidada del modelo base y calibrador. Las variaciones por fuente y fold, incluidos resultados inferiores al promedio, se conservan en calibracion_score.json. Zancada tiene solo 84 verdes en la comprobación y su acierto es 71,4% en REHAB246 y 82,1% en UI-PRMD; es el ejercicio menos consistente.

Las medidas observables se muestran aparte y una desviación medida tiene prioridad. Los cuatro modelos todavía conservan camera_validated=false: habilitar verde no valida el acierto con cámaras. La probabilidad corresponde a referencia, no estima seguridad ni lesiones. Los clips estáticos, las articulaciones ocultas y los datos fuera del rango del modelo generan un motivo específico de amarillo. El botón de diagnóstico descarga esos motivos, el color, la calidad de la secuencia y el resultado del detector, sin video original.

El lado Automático escoge la extremidad con más movimiento estimado en 3D. Puedes seleccionarlo manualmente si el ejercicio lo requiere. No se verifica toda la técnica ni se estima una lesión futura.

## Revisiones y siguiente entrenamiento

El formulario requiere revisión humana y permiso. Usa un alias estable por persona. Los errores admiten varias etiquetas; no inventes subtipos para los que no exista revisión. Una predicción no es una etiqueta real.

Las revisiones quedan separadas por sesión. Descarga moveai_videos_revisados.zip después de guardar; contiene coordenadas y etiquetas, no videos originales. Conserva cada paquete, porque cerrar la sesión o reiniciar el servidor puede borrar los datos locales. Los modelos incluidos en GitHub permanecen; los datos nuevos no se suben automáticamente a GitHub o Drive.

Importa los paquetes en el Colab mediante IMPORTAR_REVISIONES_ZIP y ejecuta el entrenamiento con ACTUALIZAR_MODELOS=True. Para varios paquetes puedes importar cada uno antes de entrenar. Reemplaza luego modelos_rapidos/ en GitHub con los nuevos archivos. La app no aprende automáticamente después de pulsar Guardar. El train.py de este paquete recalcula el calibrador después de exportar un modelo. Si usas un Colab con un train.py anterior, ejecuta calibrate_scores.py sobre la carpeta exportada. Una calibración de otros pesos se rechaza y nunca habilita verde.

## Reproducir la calibración sin entrenar

Con requirements_training.txt instalado, ejecuta:

```bash
python calibrate_scores.py --models modelos_rapidos
python -m unittest discover -s tests -v
```

Se incluyen predicciones_oof.npz, sin coordenadas ni videos, para reproducir esta comprobación. calibracion_score.json contiene los parámetros, hashes de vinculación al modelo y auditorías; evaluacion_semaforo.npz conserva los scores comprobados por fold. Streamlit solo necesita modelo.json, calibracion_score.json y los módulos de inferencia. No necesita scikit-learn al iniciar.

## Ejecución local

Usa Python 3.12, instala requirements.txt y en Linux también las bibliotecas de packages.txt. Crea .streamlit/secrets.toml con app_password. Ejecuta:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

requirements_training.txt añade las dependencias de entrenamiento y pruebas. La app desplegada no las necesita. Se ejecutaron la calibración con scores reales y 38 pruebas de geometría, decisiones, vinculación de calibradores y sesiones. La ejecución actual de Streamlit Community Cloud y la prueba con el video original del usuario quedan pendientes: este paquete no fue desplegado en tu cuenta y ese MP4 no está disponible aquí.
