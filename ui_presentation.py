"""Presentación pública; no interviene en la evaluación del movimiento."""
import streamlit as st
from feedback_presentation import count_labels

UI_REVISION = "2026.11.03-r12-publica"
CAPTURE_GUIDES = {
    "squat": "Sentadilla: graba de lado para revisar el recorrido; de frente para observar rodillas y pies. Incluye el cuerpo completo.",
    "elbow_flexion": "Curl de bíceps: graba de lado o en diagonal, con el hombro, el codo, la muñeca y la cadera del brazo activo visibles.",
    "shoulder_abduction": "Elevación lateral: graba de frente, con hombros, brazos y caderas visibles. Evita que las manos salgan del encuadre al elevar los brazos.",
    "inline_lunge": "Zancada: incluye el cuerpo completo y ambos pies, con una vista de lado o en diagonal que permita observar el recorrido de las piernas.",
}

def reset_workspace():
    workspace = st.session_state.pop("workspace", None)
    if workspace is not None:
        workspace.close()
    st.session_state.pop("review_export", None)
    st.session_state["upload_generation"] = st.session_state.get("upload_generation", 0) + 1

def show_counts(result):
    labels = count_labels(result)
    cols = st.columns(3)
    for col, label, color in zip(cols, labels, ("verde", "rojo", "amarillo")):
        col.metric(label, result["counts"][color])
    st.caption("Resumen por repetición. El resultado general del video puede ser distinto.")

def apply_style():
    st.markdown('''<style>
    .block-container { max-width: 1280px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1 { letter-spacing: -.035em; }
    h2, h3 { letter-spacing: -.02em; }
    [data-testid="stMetric"] { border: 1px solid rgba(148,163,184,.28); border-radius: 12px; padding: .8rem; }
    [data-testid="stFileUploader"] { border: 1px dashed rgba(148,163,184,.5); border-radius: 12px; padding: .6rem; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 10px; min-height: 44px; }
    @media (max-width: 700px) {
      .block-container { padding: 1.2rem 1rem 2rem; }
      [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 1rem; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: min(100%, 320px); flex: 1 1 320px; }
    }
    </style>''', unsafe_allow_html=True)
