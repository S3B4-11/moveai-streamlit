"""Inferencia neuronal optativa. Mantiene el contrato de la aplicación RGB 4."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np

@lru_cache(maxsize=4)
def cached(folder,stamp):
    from neural_models import load_bundle
    return load_bundle(folder)

def predict(sample,exercise,root=None,quality_context=None):
    try:
        base=Path(root or Path(__file__).parent/"neural_modelos")/exercise
        if not (base/"model.json").exists():return dict(available=False,reason="Sin red neuronal exportada para este ejercicio.")
        meta=json.loads((base/"model.json").read_text())
        if not meta.get("approved",False):return dict(available=False,reason="La red no cumple el filtro de piloto interno.")
        from neural_features import encode
        from neural_models import logits,probability
        from camera_model import signature,features
        if meta["extractor_id"]!=sample["extractor_id"] or meta["pipeline_signature"]!=signature():return dict(available=False,reason="Extractor o pipeline incompatible.")
        _,result=features(sample,exercise)
        resolved=bool(quality_context and quality_context.get('sample') is sample and quality_context.get('frontal_projection_repaired') and quality_context['quality']['ready'])
        if not result["quality"]["ready"] and not resolved:return dict(available=False,reason="Medidas insuficientes para la red.")
        model,norm,meta=cached(str(base.resolve()),(base/"model.json").stat().st_mtime_ns)
        x,mask=encode(sample);z=norm.transform(x[None],mask[None])
        observed=mask[:,:,:5]
        if observed.any() and (abs(z[0,:,:,:5][observed])>8).mean()>.25:return dict(available=False,reason="Pose alejada del dominio de entrenamiento.")
        p=float(probability(logits(model,z),meta["calibration"])[0])
        return dict(available=True,p_error=p,p_correct=1-p,family=meta["family"],metrics=meta["metrics"],
            scope=f"Red {meta['family'].upper()}. {meta['scope']}")
    except (ImportError,OSError,ValueError,KeyError,RuntimeError) as exc:
        return dict(available=False,reason=f"Red no disponible: {type(exc).__name__}: {exc}")
