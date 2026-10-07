"""MOVEAI para Streamlit Community Cloud. No entrena ni descarga datasets al iniciar."""
from pathlib import Path
import hmac
import logging
import streamlit as st

from geometry import EXERCISES, TAGS
from moveai_core import evaluate, markdown_result
from feedback import save_review, export_reviews
from web_session import SessionWorkspace, model_summary

st.set_page_config(page_title='MOVEAI · Revisa tu técnica', page_icon='🏋️', layout='wide')


def authenticate():
    try:
        password = st.secrets.get('app_password', '')
    except FileNotFoundError:
        password = ''
    if not isinstance(password, str) or len(password) < 8:
        st.title('MOVEAI')
        st.info('Falta configurar la contraseña de acceso. En Streamlit abre Settings → Secrets '
                'y añade app_password con una contraseña de al menos 8 caracteres.')
        st.stop()
    if st.session_state.get('authenticated'):
        return
    st.title('MOVEAI')
    st.write('Acceso para el equipo y sus amigos.')
    with st.form('login'):
        entered = st.text_input('Contraseña', type='password', key='login_password')
        submitted = st.form_submit_button('Entrar', type='primary')
    if submitted:
        if hmac.compare_digest(entered.encode(), password.encode()):
            st.session_state.authenticated = True
            st.session_state.pop('login_password', None)
            st.rerun()
        else:
            st.error('Contraseña incorrecta.')
    st.stop()


authenticate()
if 'workspace' not in st.session_state:
    st.session_state.workspace = SessionWorkspace()
workspace = st.session_state.workspace

with st.sidebar:
    st.title('MOVEAI')
    st.caption('Una repetición. Medidas visibles. Recomendaciones concretas.')
    st.write('Descarga tus revisiones antes de salir. No se conservan si la sesión o el servidor se reinician.')
    if st.button('Cerrar sesión'):
        workspace.close()
        st.session_state.clear()
        st.rerun()

st.title('Revisa tu técnica')
st.write('Sube una repetición completa, con cámara fija, cuerpo visible y una sola persona.')
controls, output = st.columns([1, 1.4], gap='large')

with controls:
    exercise = st.selectbox('Ejercicio', list(EXERCISES), format_func=EXERCISES.get)
    default_view = 'frente' if exercise == 'shoulder_abduction' else 'perfil'
    views = ['perfil', 'frente']
    view = st.radio('Vista de la cámara', views, index=views.index(default_view),
                    key='view_' + exercise, horizontal=True)
    side = st.radio('Brazo activo / pierna delantera en zancada', ['derecha', 'izquierda'],
                    key='side_' + exercise, horizontal=True)
    default_target = {'squat': 80, 'inline_lunge': 85, 'shoulder_abduction': 90, 'elbow_flexion': 120}[exercise]
    target = st.slider('Recorrido objetivo (grados)', 40, 170, default_target, 5, key='target_' + exercise)
    st.caption('El objetivo depende del ejercicio que quieres hacer. Elevación lateral: 90° por defecto. '
               'Para hombro usa vista frontal; para medir rodilla o codo, perfil. '
               'Los ángulos se estiman desde el video.')
    upload = st.file_uploader('Video de hasta 60 segundos', type=['mp4', 'mov', 'avi', 'webm', 'm4v'])
    try:
        workspace.set_input(upload.getvalue() if upload else None,
                            Path(upload.name).suffix if upload else '.mp4', exercise, view, side, target)
    except ValueError as error:
        workspace.clear_video()
        st.error(str(error))
    if upload:
        st.video(upload)
    analyze = st.button('Analizar repetición', type='primary', disabled=workspace.video is None,
                        use_container_width=True)
    if analyze:
        workspace.result = None
        with st.spinner('Midiendo la repetición…'):
            try:
                workspace.result = evaluate(workspace.video, exercise, view, side, target)
            except ValueError as error:
                st.error(str(error))
            except Exception:
                logging.exception('MOVEAI: fallo durante la evaluación')
                st.error('No se pudo procesar el video. Prueba un MP4 corto. '
                         'Si vuelve a ocurrir, revisa los registros de Streamlit.')

with output:
    result = workspace.result
    if result:
        st.markdown(markdown_result(result))
        if result['sample'].get('thumbnail') is not None:
            st.image(result['sample']['thumbnail'], caption='Fotograma de referencia', use_container_width=True)
        with st.expander('Comparación con el detector aprendido'):
            learned = result['learned']
            st.write(learned.get('reason', 'Sin detector disponible.'))
            if learned.get('available'):
                st.write('Señal del detector: ' + ('posible ejecución incorrecta' if learned['error'] else 'posible ejecución correcta'))
                st.caption('Esta señal no certifica la técnica ni es un porcentaje de seguridad. '
                           'Si no hay validación de cámara, no decide el resultado principal.')
                st.json({k: v for k, v in learned.items() if k not in ['p_error', 'reviewed_subtypes']})
    else:
        st.info('El resultado aparecerá aquí después de analizar el video.')
    with st.expander('Qué puede observar MOVEAI'):
        st.write('Recorrido articular, algunas compensaciones del tronco y, en vista frontal, '
                 'algunas desviaciones de rodilla. Puede dejar criterios sin medir si la vista '
                 'o la visibilidad no permiten evaluarlos. Estar en rango no verifica toda la técnica.')

st.divider()
review_tab, model_tab = st.tabs(['Revisión humana y datos', 'Resultados del entrenamiento'])

with review_tab:
    st.subheader('Ayuda a mejorar el próximo modelo')
    st.write('Una persona que pueda evaluar la técnica debe revisar el video. '
             'Usa el mismo alias en todos los videos de cada persona. '
             'La etiqueta se decide mirando el video, sin copiar la predicción de MOVEAI.')
    result = workspace.result
    if result and result['sample'].get('trainable'):
        with st.form('review_' + exercise):
            person = st.text_input('Alias de la persona', key='person_' + exercise)
            reviewer = st.text_input('Quién revisó la técnica', key='reviewer_' + exercise)
            label = st.selectbox('Etiqueta revisada', ['Selecciona una etiqueta', 'Correcta', 'Incorrecta'])
            tags = st.multiselect('Errores observados (opcional, permite varios)', TAGS[exercise],
                                  format_func=lambda t: t.replace('_', ' ').capitalize())
            st.caption('Para una repetición correcta deja los errores vacíos. '
                       'Para una incorrecta, puedes dejarlos vacíos si no conoces el subtipo.')
            consent = st.checkbox('La persona acepta guardar sus coordenadas y etiquetas para este proyecto')
            reviewed = st.checkbox('Revisé el video y no copié la predicción como etiqueta')
            save = st.form_submit_button('Guardar revisión')
        if save:
            try:
                if label not in ('Correcta', 'Incorrecta'):
                    raise ValueError('Selecciona la etiqueta que confirmó el revisor.')
                save_review(workspace.video, result['sample'], exercise, person, label == 'Correcta',
                            tags, reviewer, consent, reviewed, root=workspace.reviews)
                archive = Path(export_reviews(workspace.reviews))
                st.session_state.review_export = archive.read_bytes()
                st.success('Revisión guardada en esta sesión. Descarga el paquete para conservarla. '
                           'El modelo actual no cambia hasta el próximo entrenamiento.')
            except ValueError as error:
                st.error(str(error))
    else:
        st.caption('Analiza un video con articulaciones visibles para habilitar la revisión.')
    if st.session_state.get('review_export'):
        st.download_button('Descargar mis revisiones de esta sesión', st.session_state.review_export,
                           file_name='moveai_videos_revisados.zip', mime='application/zip', on_click='ignore')
    st.caption('El paquete contiene coordenadas, alias y etiquetas; no incluye el video original. '
               'Solo incluye tus revisiones de esta sesión. Impórtalo en el Colab para el siguiente entrenamiento.')

with model_tab:
    st.dataframe(model_summary(Path(__file__).parent / 'modelos_rapidos'), hide_index=True,
                 use_container_width=True)
    st.caption('Resultados en datasets de referencia con separación de personas. '
               'No representan el acierto con tus videos. Cada vista y objetivo necesita evaluación con cámara.')

st.caption('MOVEAI ofrece orientación sobre los criterios observados. No predice lesiones.')
