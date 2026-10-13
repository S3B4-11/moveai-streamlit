"""MOVEAI: resultado comprensible, por repetición."""
import hmac
import json
import logging
import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st
from geometry import EXERCISES,TARGETS
from web_session import SessionWorkspace
from moveai_core import evaluate,diagnostic
from preview import annotated
from video_overlay import render_video
from feedback import save_review,export_reviews
from feedback_presentation import APP_REVISION, POLICY_VERSION, repetition_label, repetition_color, count_labels
from pose_selection import selection_preview

RESULT_LABELS={"verde":"Ejercicio realizado correctamente","rojo":"Ejercicio mal ejecutado","amarillo":"No se pudo evaluar"}
REP_LABELS={"verde":"Correcta","rojo":"A corregir","amarillo":"Sin evaluar"}

def result_explanation(result):
    if result.get("feedback_version"):return result["reason"]
    if result["color"]=="verde":
        return "El movimiento cumple los criterios revisados en este video."
    if result["color"]=="rojo":
        return "Se detectaron aspectos de la técnica que necesitan revisión."
    reps=result["repetitions"]
    if not any(r["complete"] for r in reps):
        return "Graba una repetición completa, desde el inicio hasta el retorno."
    if result.get("decision_source") in ("modelo_incierto","evidencia_en_conflicto"):
        return "No hay una respuesta clara con esta grabación. Prueba otro video con el cuerpo visible."
    if any(not r["complete"] for r in reps):
        return "Hay una repetición incompleta. Puedes ver abajo el resultado de las demás."
    return "La grabación no permite dar un resultado claro. Prueba otro video con el cuerpo visible."

st.set_page_config(page_title="MOVEAI · Tu técnica",page_icon="🏋️",layout="wide")
try:password=st.secrets.get("app_password","")
except FileNotFoundError:password=""
if not isinstance(password,str) or len(password)<8:
    st.info("Configura app_password en Settings → Secrets, con 8 caracteres o más.");st.stop()
if not st.session_state.get("authenticated"):
    st.title("MOVEAI");st.caption("Revisa tu movimiento, repetición por repetición.")
    with st.form("login"):
        entered=st.text_input("Contraseña",type="password");sent=st.form_submit_button("Entrar",type="primary")
    if sent:
        if hmac.compare_digest(entered.encode(),password.encode()):st.session_state.authenticated=True;st.rerun()
        else:st.error("Contraseña incorrecta.")
    st.stop()
if "workspace" not in st.session_state:st.session_state.workspace=SessionWorkspace()
if not hasattr(st.session_state.workspace,"normalized_video"):
    st.session_state.workspace.close();st.session_state.workspace=SessionWorkspace()
if not getattr(st.session_state.workspace,'supports_person_selection',False):
    st.session_state.workspace=SessionWorkspace.upgrade(st.session_state.workspace)
workspace=st.session_state.workspace
if st.session_state.get('_app_revision')!=APP_REVISION:
    workspace.clear_result()
    st.session_state['_app_revision']=APP_REVISION
with st.sidebar:
    st.title("MOVEAI");st.caption("Revisa tu técnica")
    st.caption(f"Versión {APP_REVISION}")
    st.write("Graba el cuerpo y la extremidad activa durante el inicio y el retorno. Hasta 60 segundos.")
    st.caption("Sentadilla: de frente para revisar rodillas y pies; de lado para revisar el recorrido. Curl: brazo activo visible. Elevación lateral: frontal.")
    if st.button("Cerrar sesión"):workspace.close();st.session_state.clear();st.rerun()
st.title("Revisa tu técnica")
st.write("Sube tu video y revisa el resultado de cada repetición.")
controls,output=st.columns([1,1.5],gap="large")
with controls:
    exercise=st.selectbox("Ejercicio",list(EXERCISES),format_func=EXERCISES.get)
    upload=st.file_uploader("Video",type=["mp4","mov","avi","webm","m4v"])
    uploaded_data=upload.getvalue() if upload else None
    video_key='r8_'+(hashlib.sha256(uploaded_data).hexdigest()[:16] if uploaded_data else 'empty')+'_'+exercise
    with st.expander("Ajustar ejercicio y toma"):
        target=st.slider("Recorrido objetivo (grados)",40,170,TARGETS[exercise],5,key=video_key+'_target')
        side=st.selectbox("Extremidad",["automatica","izquierda","derecha"],format_func=lambda x:"Automática: lados activos" if x=="automatica" else x.capitalize()+" de la persona",key=video_key+'_side')
        st.caption("El resultado indica también el lado en la imagen para evitar confusiones.")
        rotation=st.selectbox("Orientación del video",["auto",0,90,-90,180],format_func=lambda x:{"auto":"Automática",0:"Sin giro adicional",90:"90° a la derecha",-90:"90° a la izquierda",180:"180°"}[x],key=video_key+'_rotation')
        choose_person=st.checkbox('Elegir persona (si aparecen varias)',key=video_key+'_choose')
        person_target=None
        if choose_person:
            st.caption('Coloca la marca en el centro del tronco de la persona que quieres revisar.')
            px=st.slider('Posición horizontal (%)',5,95,50,key=video_key+'_px')
            py=st.slider('Posición vertical (%)',5,95,40,key=video_key+'_py')
            person_target=[px/100.,py/100.]
    try:workspace.set_input(uploaded_data,Path(upload.name).suffix if upload else ".mp4",exercise,side,target,rotation,person_target)
    except ValueError as error:st.error(str(error))
    if upload:st.video(upload)
    if workspace.video and person_target is not None:
        selection=selection_preview(workspace.video,person_target)
        if selection is not None:st.image(selection,caption='Marca de la persona seleccionada',use_container_width=True)
    if st.button("Analizar video",type="primary",disabled=workspace.video is None,use_container_width=True):
        workspace.clear_result()
        with st.spinner("Analizando tu técnica…"):
            try:
                workspace.result=evaluate(workspace.video,exercise,side,target,rotation=rotation,person_target=person_target)
                if workspace.result.get('feedback_version')!=POLICY_VERSION:
                    workspace.clear_result()
                    raise ValueError('El motor cargado pertenece a una versión anterior. Reinicia la app con Reboot app y vuelve a analizar.')
                analysis_path=Path(workspace.result.get("analysis_video",workspace.video))
                if analysis_path!=workspace.video:workspace.normalized_video=analysis_path
            except ValueError as error:st.error(str(error))
            except Exception:
                logging.exception("MOVEAI RGB: evaluación")
                st.error("No se pudo procesar el video. Revisa los registros de la app.")
        if workspace.result:
            with st.spinner("Preparando tu video con el análisis…"):
                try:workspace.rendered_video=render_video(workspace.result.get("analysis_video",workspace.video),workspace.result,workspace.root/"analizado.mp4")
                except Exception:
                    logging.exception("MOVEAI RGB: video anotado")
                    workspace.render_error="No se pudo preparar el video anotado. El resultado del análisis sigue disponible."
with output:
    result=workspace.result
    if not result:st.info("Tu resultado aparecerá aquí.")
    else:
        {"verde":st.success,"rojo":st.error,"amarillo":st.warning}[result["color"]](result["title"] if result.get("feedback_version") else RESULT_LABELS[result["color"]])
        st.write(result_explanation(result))
        if result["color"]=="rojo":st.write("Pide a un profesional que revise tu técnica.")
        reported=set()
        for error in result.get("errors",[]):
            identity=(error["key"],error["name"])
            if identity in reported:continue
            reported.add(identity)
            st.write("**Qué corregir:** "+error["name"])
            st.write(error["advice"])
            st.write("**Qué puede provocar:** "+error["consequence"])
            if error.get('scope')=='postura':st.caption(f"Postura observada entre {error['start_time']:.1f} y {error['end_time']:.1f} s; no se cuenta como una repetición.")
        if workspace.rendered_video and workspace.rendered_video.exists():
            st.video(str(workspace.rendered_video))
            st.caption("Verde: correcta · Rojo: a corregir · Gris: sin fallo localizado · Amarillo: tramo sin evaluar.")
            st.download_button("Descargar video analizado",workspace.rendered_video.read_bytes(),file_name="moveai_video_analizado.mp4",mime="video/mp4",on_click="ignore")
        elif workspace.render_error:st.info(workspace.render_error)
        reps=result["repetitions"]
        if not reps and side!="automatica":st.info("La extremidad seleccionada no mostró un ciclo evaluable. Prueba Automática para seguir el lado que realiza el ejercicio.")
        with st.expander("Ver repeticiones"):
            if reps:
                preferred=next((i for i,r in enumerate(reps) if r["complete"] and r["color"]==result["color"]),
                               next((i for i,r in enumerate(reps) if r["complete"]),0))
                chosen=st.selectbox("Ver una repetición",range(len(reps)),index=preferred,
                                   format_func=lambda i:f'Repetición {i+1} · {repetition_label(reps[i],result)}')
                rep=reps[chosen]
                {"verde":st.success,"rojo":st.error,"amarillo":st.warning,"neutro":st.info}[repetition_color(rep,result)](f'Repetición {chosen+1}: {repetition_label(rep,result)}')
                if rep['color']=='verde' and result['color']!='verde':st.caption('Los criterios revisados cumplen en esta repetición. Eso no confirma todos los aspectos de la técnica ni localiza el resultado del clasificador.')
                if rep["color"]!="verde":st.write(rep["reason"])
                for x in rep["checks"]:
                    if x["status"] in ["fuera_de_rango","limite"]:st.write("• "+x["advice"])
        with st.expander("Ver detalles del análisis"):
            labels=count_labels(result)
            a,b,c=st.columns(3);a.metric(labels[0],result['counts']['verde']);b.metric(labels[1],result['counts']['rojo']);c.metric(labels[2],result['counts']['amarillo'])
            st.caption('Los contadores describen criterios por repetición. El clasificador califica el video completo.')
            for observation in result.get('posture_observations',[]):
                names='; '.join(c['name']+(' · a corregir' if c['status']=='fuera_de_rango' else ' · sin fallo medido') for c in observation['checks'])
                st.write(f"Postura {observation['start_time']:.1f}–{observation['end_time']:.1f} s: {names}")
            st.caption(f"Versión del análisis: {result.get('app_revision','anterior')}")
            st.write(result["reason"])
            if result["criterion_score"] is not None:
                st.metric("Cumplimiento de criterios medidos",f'{result["criterion_score"]:.0%}')
                st.caption("Es cumplimiento de criterios; no una probabilidad de acertar.")
            if result.get("orientation_rotation"):st.caption(f'Se corrigió la orientación del video ({result["orientation_rotation"]}°).')
            if result.get("feedback_version"):st.caption("El resultado se basa en los criterios visibles y en la clasificación del video. No es un diagnóstico de lesión.")
            if result["model"]["available"]:
                st.write(f'Estimación del modelo: {result["model"]["p_correct"]:.1%} de ejecución correcta.')
                st.caption(result["model"]["scope"])
            else:st.caption(result["model"]["reason"])
            if reps:
                preview=annotated(result["sample"],rep)
                if preview is not None:st.image(preview,caption=f'Repetición {chosen+1}',use_container_width=True)
                st.caption(f'Repetición {chosen+1} · {rep["start_time"]:.1f}–{rep["end_time"]:.1f} s · {rep["screen_side"]}')
                words={"en_rango":"Cumple","fuera_de_rango":"A corregir","limite":"Cerca del límite","no_evaluable":"Sin evaluar"}
                st.dataframe([{"Criterio":x["name"],"Medida":x["value"],"Unidad":x.get("unit","°"),"Referencia":("≥ " if x["direction"]=="min" else "≤ ")+str(x["reference"]),"Resultado":"Estimación por confirmar" if x.get('evidence_status')=='uncertain_3d' else words[x["status"]]} for x in rep["checks"]],hide_index=True,use_container_width=True)
            chart=pd.DataFrame({"Tiempo (s)":result["sample"]["times"]})
            for t in result["tracks"]:chart[t["screen_side"]]=t["series"]
            if len(chart.columns)>1:st.line_chart(chart.set_index("Tiempo (s)"),y_label="Ángulo (°)")
            st.caption("Los huecos corresponden a puntos rechazados; no se inventa un recorrido durante una oclusión.")
            st.download_button("Descargar diagnóstico",json.dumps(diagnostic(result),ensure_ascii=False,indent=2,allow_nan=False),file_name="moveai_diagnostico_rgb.json",on_click="ignore")
st.divider()
with st.expander("Ayudar a mejorar MOVEAI (opcional)"):
    result=workspace.result
    st.write("Revisa el video y guarda la etiqueta. Se incorpora al siguiente entrenamiento; la app no usa sus propias predicciones como respuestas correctas.")
    if result and result["sample"]["trainable"]:
        with st.form("review"):
            subject=st.text_input("Alias de la persona (siempre el mismo)")
            reviewer=st.text_input("Quién revisó")
            label=st.selectbox("Etiqueta revisada",["Selecciona","Correcta","Incorrecta"])
            consent=st.checkbox("La persona acepta guardar sus coordenadas para el proyecto")
            reviewed=st.checkbox("Revisé el video completo")
            sent=st.form_submit_button("Guardar revisión")
        if sent:
            try:
                if label=="Selecciona":raise ValueError("Selecciona una etiqueta.")
                save_review(workspace.video,result["sample"],exercise,subject,label=="Correcta",reviewer,consent,reviewed,workspace.reviews)
                st.session_state.review_export=export_reviews(workspace.reviews).read_bytes();st.success("Guardada. Descarga el paquete para conservarla.")
            except ValueError as error:st.error(str(error))
    else:st.caption("Primero analiza un video con una lectura suficiente.")
    if st.session_state.get("review_export"):st.download_button("Descargar revisiones",st.session_state.review_export,file_name="moveai_revisiones_rgb.zip",on_click="ignore")
st.caption("MOVEAI revisa los movimientos visibles en tu video.")
