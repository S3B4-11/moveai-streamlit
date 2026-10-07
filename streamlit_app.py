"""MOVEAI: resultado comprensible, por repetición."""
import hmac
import json
import logging
from pathlib import Path
import pandas as pd
import streamlit as st
from geometry import EXERCISES,TARGETS
from web_session import SessionWorkspace
from moveai_core import evaluate,diagnostic
from preview import annotated
from feedback import save_review,export_reviews

RESULT_LABELS={"verde":"Ejecución correcta","rojo":"Necesita corrección","amarillo":"No se pudo evaluar"}
REP_LABELS={"verde":"Correcta","rojo":"A corregir","amarillo":"Sin evaluar"}

def result_explanation(result):
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
workspace=st.session_state.workspace
with st.sidebar:
    st.title("MOVEAI");st.caption("Revisa tu técnica")
    st.write("Graba el cuerpo y la extremidad activa durante el inicio y el retorno. Hasta 60 segundos.")
    st.caption("Curl: frontal o diagonal con el brazo activo visible. Sentadilla: lateral. Elevación lateral: frontal.")
    if st.button("Cerrar sesión"):workspace.close();st.session_state.clear();st.rerun()
st.title("Revisa tu técnica")
st.write("Sube tu video y revisa el resultado de cada repetición.")
controls,output=st.columns([1,1.5],gap="large")
with controls:
    exercise=st.selectbox("Ejercicio",list(EXERCISES),format_func=EXERCISES.get)
    with st.expander("Ajustar ejercicio y toma"):
        target=st.slider("Recorrido objetivo (grados)",40,170,TARGETS[exercise],5,key="target_"+exercise)
        side=st.selectbox("Extremidad",["automatica","izquierda","derecha"],format_func=lambda x:"Automática: lados activos" if x=="automatica" else x.capitalize()+" de la persona")
        st.caption("El resultado indica también el lado en la imagen para evitar confusiones.")
    upload=st.file_uploader("Video",type=["mp4","mov","avi","webm","m4v"])
    try:workspace.set_input(upload.getvalue() if upload else None,Path(upload.name).suffix if upload else ".mp4",exercise,side,target)
    except ValueError as error:st.error(str(error))
    if upload:st.video(upload)
    if st.button("Analizar video",type="primary",disabled=workspace.video is None,use_container_width=True):
        workspace.result=None
        with st.spinner("Analizando tu técnica…"):
            try:workspace.result=evaluate(workspace.video,exercise,side,target)
            except ValueError as error:st.error(str(error))
            except Exception:
                logging.exception("MOVEAI RGB: evaluación")
                st.error("No se pudo procesar el video. Revisa los registros de la app.")
with output:
    result=workspace.result
    if not result:st.info("Tu resultado aparecerá aquí.")
    else:
        {"verde":st.success,"rojo":st.error,"amarillo":st.warning}[result["color"]](RESULT_LABELS[result["color"]])
        st.write(result_explanation(result))
        if result["color"]=="rojo":st.write("Pide a un profesional que revise tu técnica.")
        st.caption("Resultado según los criterios visibles de esta grabación.")
        a,b,c=st.columns(3);a.metric("Correctas",result["counts"]["verde"]);b.metric("A corregir",result["counts"]["rojo"]);c.metric("Sin evaluar",result["counts"]["amarillo"])
        reps=result["repetitions"]
        if not reps and side!="automatica":st.info("La extremidad seleccionada no mostró un ciclo evaluable. Prueba Automática para seguir el lado que realiza el ejercicio.")
        if reps:
            preferred=next((i for i,r in enumerate(reps) if r["complete"] and r["color"]==result["color"]),
                           next((i for i,r in enumerate(reps) if r["complete"]),0))
            chosen=st.selectbox("Ver una repetición",range(len(reps)),index=preferred,
                               format_func=lambda i:f'Repetición {i+1} · {REP_LABELS[reps[i]["color"]]}')
            rep=reps[chosen]
            {"verde":st.success,"rojo":st.error,"amarillo":st.warning}[rep["color"]](f'Repetición {chosen+1}: {REP_LABELS[rep["color"]]}')
            if rep["color"]!="verde":st.write(rep["reason"])
            preview=annotated(result["sample"],rep)
            if preview is not None:st.image(preview,caption=f'Repetición {chosen+1}',use_container_width=True)
            for x in rep["checks"]:
                if x["status"] in ["fuera_de_rango","limite"]:st.write("• "+x["advice"])
        with st.expander("Ver detalles del análisis"):
            st.write(result["reason"])
            if result["criterion_score"] is not None:
                st.metric("Cumplimiento de criterios medidos",f'{result["criterion_score"]:.0%}')
                st.caption("Es cumplimiento de criterios; no una probabilidad de acertar.")
            if result["model"]["available"]:
                st.write(f'Estimación del modelo: {result["model"]["p_correct"]:.1%} de ejecución correcta.')
                st.caption(result["model"]["scope"])
            else:st.caption(result["model"]["reason"])
            if reps:
                st.caption(f'Repetición {chosen+1} · {rep["start_time"]:.1f}–{rep["end_time"]:.1f} s · {rep["screen_side"]}')
                words={"en_rango":"Cumple","fuera_de_rango":"A corregir","limite":"Cerca del límite","no_evaluable":"Sin evaluar"}
                st.dataframe([{"Criterio":x["name"],"Medida (°)":x["value"],"Referencia":("≥ " if x["direction"]=="min" else "≤ ")+str(x["reference"]),"Resultado":words[x["status"]]} for x in rep["checks"]],hide_index=True,use_container_width=True)
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
