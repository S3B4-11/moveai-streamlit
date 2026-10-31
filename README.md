# MOVEAI R11

Versión 2026.10.30-r11. El selector incluye Press de hombros con mancuernas, de pie o sentado, con reglas propias.

## Actualizar la app

1. Extrae el ZIP y sube su contenido a la raíz del mismo repositorio de GitHub; añade los archivos nuevos y reemplaza los de igual nombre. No subas el ZIP como un único archivo. Conserva los Secrets en Streamlit.
2. Mantén streamlit_app.py como entrada y Python 3.12. Guarda el commit y ejecuta Reboot app. Comprueba la versión en la barra lateral.
3. Selecciona Press de hombros con mancuernas. Graba preferentemente de frente o en diagonal, con hombros, codos, muñecas y tronco visibles durante el ciclo completo.

La disposición de la interfaz se conserva. La meta inicial del press es 150° de elevación del brazo (configurable 120–170°); no es el ROM total del ciclo. Flexionar los codos abajo es parte del ejercicio. Su extensión se revisa arriba, junto al recorrido, retorno y cambio del tronco. El esqueleto de una repetición a corregir permanece gris: el aviso rojo no localiza automáticamente una articulación lesionada ni convierte la flexión normal del descenso en un fallo.

## Origen de los resultados

El press se evalúa con press_analysis.py mediante criterios observables. Se prioriza 2D; world estimado sólo puede intervenir si supera los controles de visibilidad, segmentos y continuidad. No se activa el clasificador de investigación shoulder_press ni se muestra una probabilidad entrenada para press. Las firmas de entrada y los pesos de los clasificadores originales permanecen intactos. No hace falta ejecutar Colab para instalar esta corrección.

Las referencias y tolerancias son parámetros de ingeniería, pendientes de validación experta e instrumental. No constituyen umbrales clínicos. El cumplimiento de criterios no es la probabilidad de acierto, ni una certificación de toda la técnica. Los clips cortados y las oclusiones conservan su motivo.

## Pruebas reproducibles

- python -m unittest test_press test_moveai test_feedback test_video_overlay test_person_posture test_general_flow test_repetition_quality
- python diagnose_video.py video.mp4 --exercise shoulder_press --output prueba_press
- python verify_press_dataset.py --videos /ruta/videos --cache /ruta/cache --output resultados.json

press_validation.json conserva los resultados de la prueba de desarrollo con los 150 videos de hombro de Mendeley V3 y las discrepancias. No es una cohorte independiente ni valida lesiones, ángulos o segmentación por fotograma. No se eligieron umbrales consultando el resultado de cada video al cargarlo. Los videos y sus etiquetas no se distribuyen en este ZIP.

Las revisiones de press no se envían al procedimiento de entrenamiento anterior. Integrar un clasificador nuevo requiere un protocolo compatible, etiquetas revisadas y evaluación por participante. El aprendizaje no ocurre automáticamente a partir de la propia predicción de la app.

Fuente de los videos: Sai Prashanth et al., Mendeley Data V3, DOI 10.17632/kgbb3yn47p.3, CC BY 4.0. https://data.mendeley.com/datasets/kgbb3yn47p/3


Actualización conservadora R11: se requieren al menos dos muestras medidas en inicio, fase superior y retorno, además de conservar visibilidad >=0,55 y lectura de al menos el 80% de la repetición. Una desviación fuera de tolerancia debe mantenerse al menos 0,25 s en muestras consecutivas para considerarse un fallo firme. Una desviación breve queda registrada para confirmar y nunca se convierte por ese motivo en verde. La app muestra el número de ciclos revisados y sus resultados aunque no califique el video completo. No se aplica voto mayoritario ni se ocultan ciclos sin lectura.

Comparación de desarrollo sobre los mismos 150 videos: las falsas alarmas frente a good/bad bajaron de 18 a 12 y las aprobaciones incorrectas de 6 a 5, sin nuevos casos de esos errores. Los videos sin conclusión global aumentaron de 65 a 81; 47 de estos 81 contienen resultados locales evaluables. Se afirma menos para evitar errores. Estas cifras no prueban una mejora de generalización ni certifican lesiones o mediciones angulares. La alternativa de conteo que añadía un nuevo caso de cada error fue descartada.
