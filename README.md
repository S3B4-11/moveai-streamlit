# MOVEAI en Streamlit

Este paquete usa los cuatro modelos del ZIP MOVEAI_privado_actualizado que exportaste en Colab. No vuelve a entrenar ni descarga los datasets al abrir la app. Incluye el detector de pose para evitar una descarga al arrancar.

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

Los resultados principales son reglas sobre medidas visibles. Los cuatro modelos aprendidos tienen camera_validated=false y solo se muestran como comparación hasta validar cámara. No se verifica toda la técnica ni se estima una lesión futura.

## Revisiones y siguiente entrenamiento

El formulario requiere revisión humana y permiso. Usa un alias estable por persona. Los errores admiten varias etiquetas; no inventes subtipos para los que no exista revisión. Una predicción no es una etiqueta real.

Las revisiones quedan separadas por sesión. Descarga moveai_videos_revisados.zip después de guardar; contiene coordenadas y etiquetas, no videos originales. Conserva cada paquete, porque cerrar la sesión o reiniciar el servidor puede borrar los datos locales. Los modelos incluidos en GitHub permanecen; los datos nuevos no se suben automáticamente a GitHub o Drive.

Importa los paquetes en el Colab mediante IMPORTAR_REVISIONES_ZIP y ejecuta el entrenamiento con ACTUALIZAR_MODELOS=True. Para varios paquetes puedes importar cada uno antes de entrenar. Reemplaza luego modelos_rapidos/ en GitHub con los nuevos archivos. La app no aprende automáticamente después de pulsar Guardar.

## Ejecución local

Usa Python 3.12, instala requirements.txt y en Linux también las bibliotecas de packages.txt. Crea .streamlit/secrets.toml con app_password. Ejecuta:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

requirements_training.txt añade las dependencias de entrenamiento y pruebas. La app desplegada no las necesita. La ejecución de Streamlit Community Cloud queda pendiente: este paquete se preparó localmente y no fue desplegado en tu cuenta.
