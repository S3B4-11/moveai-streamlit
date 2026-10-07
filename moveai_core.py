"""API pública: video, repeticiones y evidencia de cámara."""
from analysis import analyze
from pose import extract_video
from camera_model import predict,fuse,hard_failure

def evaluate(path,exercise,side="automatica",target=None):
    result=analyze(extract_video(path),exercise,side,target)
    model=predict(result["sample"],exercise);result["model"]=model
    if model["available"]:
        color,source=fuse(result["color"],model["p_correct"],result["quality"]["ready"],hard_failure(result))
        result.update(color=color,decision_source=source)
        if source=="modelo_rgb":
            result.update(title="Ejecución probablemente correcta" if color=="verde" else "Revisar la ejecución",
                          reason="El detector RGB supera 75% para esta clase. El recorrido objetivo y los criterios por repetición se muestran por separado.")
        elif source=="evidencia_en_conflicto":
            result.update(title="El modelo y las medidas necesitan revisión",reason="Hay evidencia contradictoria; no se convierte en una afirmación segura de técnica incorrecta.")
        elif source=="modelo_incierto":
            result.update(title="El detector necesita revisión",reason="No alcanza 75% de correcta ni de incorrecta. Revisa los criterios y el video.")
    return result

def diagnostic(result):
    import math
    import numpy as np
    payload={k:v for k,v in result.items() if k not in ["sample","tracks"]}
    payload["tracks"]=[{k:v for k,v in t.items() if k not in ["series","cycle_series","valid"]} for t in result["tracks"]]
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    return clean(payload)
