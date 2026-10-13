"""Selecciona backend sin alterar widgets, estilos ni flujo de la página."""
import os
from camera_model import predict as rgb4_predict

def predict(sample,exercise,quality_context=None):
    if os.environ.get("MOVEAI_MODEL_BACKEND","auto")=="rgb4":return rgb4_predict(sample,exercise)
    from neural_runtime import predict as neural_predict
    neural=neural_predict(sample,exercise,quality_context=quality_context)
    if neural["available"]:return neural
    result=rgb4_predict(sample,exercise)
    result["neural_status"]=neural.get("reason")
    return result
