# MOVEAI RGB 4

Revisión de recorrido y compensaciones visibles, por repetición. Extremidad automática o manual y meta ajustable. Los clasificadores anteriores de sensores Kinect no se usan en la evaluación de cámara.

## Publicar en Streamlit Community Cloud

1. Extrae `MOVEAI_Streamlit.zip`. Sustituye el contenido anterior del repositorio por esta carpeta completa. El archivo `streamlit_app.py` debe estar en la raíz, junto a `requirements.txt`, `packages.txt`, `.streamlit/config.toml`, `assets/` y `camera_modelos/` si existe. Elimina los archivos antiguos antes de subir; no mezcles versiones.
2. En Streamlit crea una app con ese repositorio y `streamlit_app.py` como entrada. En Advanced settings selecciona **Python 3.12**. Cambiar la versión de Python de una app existente exige volver a desplegarla; conserva la URL y Secrets antes de hacerlo.
3. En Settings → Secrets configura `app_password = "tu-clave-de-al-menos-8-caracteres"`. Comparte esa clave con las personas del proyecto. No subas Secrets a GitHub.
4. Prueba un clip completo, de hasta 60 s y 150 MB. Para comprobar un caso dudoso descarga el diagnóstico y revisa la repetición y sus articulaciones anotadas.

## Resultado

- Verde: cumplimiento de los criterios observables, o ≥75% de correcta de un detector RGB aprobado sin una compensación clara contradictoria.
- Rojo: criterio medido fuera del margen acordado, o ≥75% de incorrecta de un detector RGB aprobado cuando las medidas no lo contradicen.
- Amarillo: toma incompleta, oclusión, cercanía al límite, incertidumbre o conflicto. No significa técnica incorrecta.
- **Cumplimiento de criterios** es una puntuación de medidas, no una probabilidad. Las metas son ajustables a la variante del ejercicio. No hay una certificación de toda la técnica ni un predictor de lesiones.

La app guarda el archivo temporal sólo en su sesión. Las revisiones se exportan como coordenadas y etiquetas humanas; descárgalas para conservarlas. No se aprende automáticamente de la propia predicción. Usa siempre el mismo alias por persona, y revisa la etiqueta con alguien que conozca la técnica.

## Datos y entrenamiento

El Colab incluido descarga automáticamente los videos de curl y sentadilla del dataset **A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form Labels**, Duddela Sai Prashanth et al., Mendeley Data V3 (2026), DOI https://doi.org/10.17632/kgbb3yn47p.3, licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Se conserva la atribución, se rechazan videos idénticos con etiquetas contradictorias y todas las vistas de una persona se mantienen en el mismo fold. Son etiquetas de forma revisadas por los autores; no se asume validación clínica.

El press de hombro de ese dataset **no** se usa como elevación lateral. Hombro y zancada conservan los criterios observables hasta disponer de suficientes revisiones RGB de ambos tipos, de al menos seis personas. Para aprender de videos propios exporta revisiones desde la app e impórtalas en el Colab. Se requieren al menos veinte clips y seis personas, con ambas clases; repetir clips no añade personas nuevas.

La validación externa por sujeto usa cinco folds. La selección y calibración usan tres repeticiones de tres folds internos, separados por sujeto. La persona 1 del dataset se reserva para depurar videos y se excluye de los resultados externos. Se calibra la media de scores fuera de muestra, del mismo tipo que la media exportada para inferencia. No se ajusta el umbral sobre los sujetos de prueba. Cada `camera_modelos/<ejercicio>/metricas.json` incluye sensibilidad, especificidad, AUC y precisión real del verde; `approved` determina si participa en la app. Un detector que no pasa el criterio permanece desactivado.

Resultados de esta entrega, sobre clips con lectura suficiente: sentadilla, 100 videos de 24 personas, accuracy balanceada 0.750, sensibilidad de error 0.800, especificidad 0.700. El semáforo completo emitió 19 verdes (15 correctos), 15 rojos (13 incorrectos) y 66 amarillos. El verde puede venir del detector ≥75% o de criterios observables cumplidos cuando el detector es incierto y se inclina por correcta; no son la misma medida. Si el detector se inclina por incorrecta, las medidas positivas no fuerzan verde. Estos resultados son de desarrollo en ese dataset, no una garantía en toda cámara. El criterio de activación para este piloto privado exige al menos 15 verdes distribuidos entre 10 personas, además de precision verde ≥0.75 y las métricas generales de detección; varias vistas de una persona no son pruebas independientes. Codo: 99 videos de 24 personas, accuracy balanceada 0.626; detector desactivado. Codo, elevación lateral y zancada muestran sólo criterios observables. No se presenta una probabilidad aprendida para esos ejercicios.

La extracción usa **MediaPipe Pose Landmarker Full**. Las características incluyen las trayectorias de inicio, pico y retorno alineadas por repetición, además de medidas y velocidad. La extracción se conserva en Drive. Cambiar criterios o probar nuevos clasificadores reutiliza las coordenadas; no obliga a repetir la extracción ni el entrenamiento LSTM. El ZIP opcional `MOVEAI_RGB_cache.zip` contiene las poses públicas ya extraídas: impórtalo con `IMPORTAR_POSES_GUARDADAS=True` cuando actualices modelos para evitar esa primera extracción. El Colab funciona con un entorno Python 3.12 independiente del Python del kernel, sin TensorFlow/Keras.

## Ejecución local

Con Python 3.12:

```
python -m pip install -r requirements_training.txt
streamlit run streamlit_app.py
python -m unittest test_moveai -q
```

Para entrenamiento instala `requirements_training.txt` y ejecuta `rgb_dataset.py`, después `train_camera.py --manifest <ruta/manifest.json> --cache <ruta/cache> --output camera_modelos --workers 2`. `diagnose_video.py <video.mp4> --exercise elbow_flexion` genera JSON y fotogramas de cada repetición. No subas datasets o cachés al repositorio de la app.
