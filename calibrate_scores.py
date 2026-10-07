"""Ajuste posterior rápido sobre predicciones OOF guardadas, sin reentrenar bosques.

La auditoría reutiliza los folds originales por persona. No es una nueva prueba
independiente: los modelos base de otros folds pueden haber visto las etiquetas
del fold evaluado. Esto se declara en el artefacto y no valida videos de cámara.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from score_calibration import CALIBRATION_VERSION, model_digest, sigmoid_probability


GREEN_THRESHOLD = .75
RED_THRESHOLD = .75
EVALUATION = ('Comprobación posterior por persona sobre scores OOF guardados; '
              'no es CV completamente anidada del calibrador y modelo base ni prueba independiente de cámara.')


def stored_folds(model, groups, y):
    partitions, seen = [], set()
    all_subjects = set(groups.tolist())
    for fold in model['folds']:
        subjects = set(fold['test_subjects'])
        if not subjects or subjects & seen or not subjects <= all_subjects:
            raise ValueError('Los folds originales no separan sujetos de forma válida.')
        seen |= subjects
        test = np.flatnonzero(np.isin(groups, sorted(subjects)))
        train = np.flatnonzero(~np.isin(groups, sorted(subjects)))
        if len(test) != fold['samples'] or len(np.unique(y[train])) != 2:
            raise ValueError('Las predicciones no corresponden a los folds del modelo.')
        partitions.append((train, test))
    if seen != all_subjects or len(partitions) < 2:
        raise ValueError('La partición no cubre todos los sujetos.')
    return partitions


def fit_sigmoid(prob, y):
    # Platt sobre la salida del bosque. Sin pesos de clase: conserva la prevalencia
    # observada de referencia. Método y cortes fijos, sin buscar un corte con test.
    model = LogisticRegression(C=1e6, max_iter=2000)
    model.fit(np.asarray(prob).reshape(-1, 1), y)
    a, b = float(model.coef_[0, 0]), float(model.intercept_[0])
    if a <= 0:
        raise ValueError('El score no separa errores en la dirección esperada.')
    np.testing.assert_allclose(sigmoid_probability(prob, a, b),
                               model.predict_proba(np.asarray(prob).reshape(-1, 1))[:, 1], atol=1e-12)
    return a, b


def color_audit(y, calibrated, groups):
    green = calibrated <= 1. - GREEN_THRESHOLD
    red = calibrated >= RED_THRESHOLD
    yellow = ~(green | red)
    n_green, n_red = int(green.sum()), int(red.sum())
    green_correct = int((y[green] == 0).sum())
    red_incorrect = int((y[red] == 1).sum())
    return {'samples': int(len(y)), 'subjects': int(len(np.unique(groups))),
            'green_samples': n_green, 'green_correct': green_correct,
            'green_precision': green_correct / n_green if n_green else None,
            'green_subjects': int(len(np.unique(groups[green]))),
            'red_samples': n_red, 'red_incorrect': red_incorrect,
            'red_precision': red_incorrect / n_red if n_red else None,
            'yellow_samples': int(yellow.sum()),
            'colored_coverage': float((green | red).mean())}


def calibrate_exercise(model_dir, oof_path=None):
    model_dir = Path(model_dir)
    path = Path(oof_path or model_dir / 'predicciones_oof.npz')
    model = json.loads((model_dir / 'modelo.json').read_text())
    with np.load(path, allow_pickle=False) as data:
        y, prob, groups, sources = [data[key].copy() for key in ('y', 'prob', 'subject', 'source')]
        threshold = data['threshold'].copy()
    if (y.ndim != 1 or any(a.shape != y.shape for a in (prob, groups, sources, threshold))
            or len(y) != model['metrics']['samples'] or set(np.unique(y)) != {0, 1}
            or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1))):
        raise ValueError('Predicciones OOF incompletas o incompatibles con el modelo.')
    original_cm = confusion_matrix(y, prob >= threshold, labels=[0, 1]).tolist()
    if (original_cm != model['metrics']['confusion_matrix']
            or not np.isclose(roc_auc_score(y, prob), model['metrics']['auc'], atol=1e-10)):
        raise ValueError('La auditoría del modelo no coincide con estas predicciones OOF.')
    calibrated = np.full(len(y), np.nan)
    fold_audits = []
    for num, (train, test) in enumerate(stored_folds(model, groups, y), 1):
        a, b = fit_sigmoid(prob[train], y[train])
        calibrated[test] = sigmoid_probability(prob[test], a, b)
        fold_audits.append(dict(color_audit(y[test], calibrated[test], groups[test]), fold=num,
                                test_subjects=sorted(np.unique(groups[test]).tolist())))
    if not np.isfinite(calibrated).all():
        raise ValueError('No se evaluaron todas las muestras.')
    audit = color_audit(y, calibrated, groups)
    audit.update(brier_raw=float(brier_score_loss(y, prob)),
                 brier_calibrated=float(brier_score_loss(y, calibrated)),
                 reference_incorrect_prevalence=float(y.mean()),
                 camera_samples=int((sources == 'camera').sum()),
                 camera_subjects=int(len(np.unique(groups[sources == 'camera']))),
                 sources={str(source): color_audit(y[sources == source], calibrated[sources == source],
                                                 groups[sources == source]) for source in np.unique(sources)},
                 folds=fold_audits)
    a, b = fit_sigmoid(prob, y)
    metadata = {'version': CALIBRATION_VERSION, 'created': datetime.now(timezone.utc).isoformat(),
                'model_fingerprint': model['fingerprint'], 'model_sha256': model_digest(model),
                'feature_schema': model['feature_schema'], 'scope': 'reference',
                'method': 'sigmoid_on_raw_forest_score', 'coefficient': a, 'intercept': b,
                'green_threshold': GREEN_THRESHOLD, 'red_threshold': RED_THRESHOLD,
                'green_supported': bool(audit['green_samples'] >= 20 and audit['green_subjects'] >= 5
                                        and audit['green_precision'] is not None
                                        and audit['green_precision'] >= GREEN_THRESHOLD),
                'evaluation': EVALUATION, 'audit': audit,
                'oof_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'limitations': ['La comprobación posterior reutiliza scores de modelos OOF y no es una nueva evaluación independiente.',
                                'La prevalencia y los sensores de referencia pueden diferir de los videos de la app.',
                                'El acierto de los verdes no es la accuracy global: se excluyen los amarillos.',
                                'La auditoría del score no incluye los controles de visibilidad, dominio o medidas 2D de la app.']}
    (model_dir / 'calibracion_score.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    np.savez_compressed(model_dir / 'evaluacion_semaforo.npz', y=y, p_error_calibrated=calibrated,
                        subject=groups, source=sources)
    return metadata


def calibrate_all(root):
    rows = []
    for directory in sorted(Path(root).iterdir()):
        if directory.is_dir() and (directory / 'modelo.json').exists() and (directory / 'predicciones_oof.npz').exists():
            metadata = calibrate_exercise(directory)
            a = metadata['audit']
            rows.append({'exercise': directory.name, 'green': a['green_samples'],
                         'green_correct': a['green_correct'], 'green_precision': a['green_precision'],
                         'yellow': a['yellow_samples'], 'red': a['red_samples'],
                         'green_supported': metadata['green_supported']})
    if not rows:
        raise ValueError('No hay modelos con predicciones_oof.npz para calibrar.')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='modelos_rapidos')
    args = parser.parse_args()
    print(json.dumps(calibrate_all(args.models), ensure_ascii=False, indent=2))
