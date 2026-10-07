"""Calibración portable de scores: la app no necesita scikit-learn."""
import hashlib
import json
from pathlib import Path
import numpy as np


CALIBRATION_VERSION = 'reference_sigmoid_v1'


def model_digest(model):
    """Vincula la calibración a los pesos y metadatos exactos del detector."""
    data = json.dumps(model, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def sigmoid_probability(raw_probability, coefficient, intercept):
    value = np.asarray(raw_probability, dtype=float)
    logit = np.clip(coefficient * value + intercept, -40, 40)
    result = 1. / (1. + np.exp(-logit))
    return float(result) if result.ndim == 0 else result


def calibrated_score(model, raw_probability, path):
    """Devuelve una probabilidad de referencia; nunca afirma validación de cámara."""
    unavailable = {'available': False, 'scope': 'reference', 'camera_calibrated': False}
    path = Path(path)
    if not path.exists():
        return dict(unavailable, reason='Falta la calibración de este modelo.')
    try:
        data = json.loads(path.read_text())
        if (data.get('version') != CALIBRATION_VERSION
                or data.get('model_fingerprint') != model.get('fingerprint')
                or data.get('model_sha256') != model_digest(model)
                or data.get('feature_schema') != model.get('feature_schema')):
            return dict(unavailable, reason='El modelo cambió: necesita una calibración que corresponda a sus pesos.')
        a, b = float(data['coefficient']), float(data['intercept'])
        green, red = float(data['green_threshold']), float(data['red_threshold'])
        if (not np.isfinite([a, b, raw_probability, green, red]).all()
                or a <= 0 or not 0 <= raw_probability <= 1
                or not .75 <= green <= 1 or not .75 <= red <= 1):
            raise ValueError('Parámetros inválidos.')
        p_error = sigmoid_probability(raw_probability, a, b)
        return {'available': True, 'scope': data['scope'], 'camera_calibrated': False,
                'p_error': p_error, 'p_correct': 1. - p_error,
                'green_threshold': green, 'red_threshold': red,
                'green_supported': bool(data['green_supported']),
                'evaluation': data['evaluation'], 'reference_audit': data['audit'],
                'reason': 'Probabilidad estimada con calibración en datasets de referencia.'}
    except (ValueError, TypeError, KeyError, OSError):
        return dict(unavailable, reason='No se pudo verificar la calibración; revisa los archivos del modelo.')
