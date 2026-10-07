"""Decisión del detector y medidas 2D independientes: una vista no oculta la predicción."""
import numpy as np
from geometry import EXERCISES, rule_assessment, signals
from pose import extract_video
from runtime_model import predict_model, predict_subtypes


def sequence_quality(sample, exercise, side='automatica'):
    """Separa calidad para el modelo 3D de disponibilidad de cada medida 2D."""
    lower = exercise in ('squat', 'inline_lunge')
    joints = [11,12,23,24,25,26,27,28] if lower else [11,12,13,14,15,16,23,24]
    visibility = np.asarray(sample['visibility'])
    world = np.asarray(sample['world'])
    visible = np.min(visibility[:, joints], axis=1) >= .55
    finite = np.isfinite(world[:, joints]).all(axis=(1,2))
    usable_frames = visible & finite
    fraction = float(usable_frames.mean())
    sig = signals(world)
    activity = 'arm' if exercise == 'shoulder_abduction' else 'knee' if lower else 'elbow'
    ranges = {}
    for label, suffix in [('izquierda','left'), ('derecha','right')]:
        series = sig[activity+'_'+suffix]
        valid = usable_frames & np.isfinite(series)
        q = np.percentile(series[valid], [10,90]) if valid.any() else [0,0]
        ranges[label] = float(q[1]-q[0])
    resolved = max(ranges, key=ranges.get) if side == 'automatica' else side
    if resolved not in ranges:
        raise ValueError('Selecciona lado automático, derecha o izquierda.')
    motion = max(ranges.values()) if exercise == 'squat' else ranges[resolved]
    reason = ''
    if sample['multiple_people']:
        reason = 'Aparece más de una persona. Graba una sola persona en cuadro.'
    elif sample['coverage'] < .8:
        reason = f"Se detectó a la persona en {sample['coverage']:.0%} de los fotogramas; se requiere 80%."
    elif fraction < .8:
        reason = f'Las articulaciones que usa el modelo son visibles en {fraction:.0%} de los fotogramas; se requiere 80%.'
    elif motion < 20:
        other = 'izquierda' if resolved == 'derecha' else 'derecha'
        reason = (f'El lado {resolved} varía solo {motion:.1f}° en la estimación 3D. '
                  + (f'El lado {other} sí se mueve; selecciona ese lado o Automático.'
                     if ranges[other] >= 20 else 'Sube una repetición con movimiento suficiente.'))
    return {'ready':not reason, 'reason':reason, 'side':resolved,
            'visible_fraction':fraction, 'movement_3d':round(motion,1),
            'movement_by_side':{k:round(v,1) for k,v in ranges.items()}}


def evaluate(path, exercise, view='perfil', side='automatica', target=None):
    if exercise not in EXERCISES:
        raise ValueError('Ejercicio desconocido.')
    if view not in ('perfil','frente'):
        raise ValueError('Selecciona la vista real de la cámara.')
    sample = extract_video(path)
    quality = sequence_quality(sample, exercise, side)
    side = quality['side']
    rules = rule_assessment(sample['image'], sample['visibility'], exercise, view, side, target)
    sample.update(view=view, side=side, target=rules['target'], trainable=quality['ready'])
    base_ready = bool(sample['coverage'] >= .8 and sample['multiple_people'] == 0)
    measurement_ready = bool(base_ready and rules['evaluable'])
    learned = predict_model(sample['world'], exercise, view=view, target=rules['target'])
    learned['reviewed_subtypes'] = predict_subtypes(sample['world'], exercise, target=rules['target'])
    # Mostrar una predicción preliminar no modifica camera_validated ni usable.
    prediction_ready = bool(quality['ready'] and learned.get('preliminary_available', False))
    if not base_ready:
        for check in rules['checks']:
            check['status'] = 'no_evaluable'
            check['reason'] = quality['reason']
    elif not measurement_ready and rules['checks'][0]['status'] != 'no_evaluable':
        rules['checks'][0]['status'] = 'no_evaluable'
        rules['checks'][0]['reason'] = rules['reason']
    deviations = [c for c in rules['checks'] if c['status'] == 'fuera_de_rango'] if base_ready else []
    source, color, decision_reason = 'quality', 'amarillo', ''
    calibration = learned.get('calibration', {})
    calibrated_ready = bool(prediction_ready and calibration.get('available'))
    borderline = [c for c in rules['checks'] if c['status'] == 'limite']
    if not quality['ready']:
        title = 'No concluyente: revisa la grabación'
        decision_reason = quality['reason']
    elif deviations:
        title = 'Revisar técnica: ' + ', '.join(c['name'].lower() for c in deviations)
        source, color = 'measures', 'rojo'
        decision_reason = 'Se observó una desviación en los criterios medidos de esta grabación.'
    elif calibrated_ready and calibration['p_error'] >= calibration['red_threshold']:
        title = 'Técnica clasificada como incorrecta: revisa la ejecución'
        source, color = 'model_calibrated', 'rojo'
        decision_reason = f"El modelo estima {calibration['p_error']:.1%} de ejecución incorrecta; el corte rojo es {calibration['red_threshold']:.0%}."
    elif borderline:
        title = 'Revisar: hay medidas cerca del límite'
        source = 'measures_borderline'
        decision_reason = ', '.join(c['name'] for c in borderline) + '. Ajusta el recorrido o revisa la repetición.'
    elif calibrated_ready and calibration['green_supported'] and calibration['p_correct'] >= calibration['green_threshold']:
        title = 'Técnica clasificada como correcta'
        source, color = 'model_calibrated', 'verde'
        decision_reason = f"El modelo estima {calibration['p_correct']:.1%} de ejecución correcta; el corte verde es {calibration['green_threshold']:.0%}."
    elif calibrated_ready:
        title = 'Revisión necesaria: confianza insuficiente para verde o rojo'
        source = 'model_uncertain'
        decision_reason = ('El resultado está entre los cortes del semáforo.' if calibration['green_supported'] else
                           'La comprobación de referencia no tiene suficientes verdes acertados para habilitar ese resultado.')
    elif prediction_ready and learned.get('near_threshold'):
        title = 'No concluyente: el modelo está cerca de su umbral'
        source = 'model_uncertain'
        decision_reason = 'Falta una calibración verificada y el score está cerca del corte anterior.'
    elif prediction_ready:
        preliminary = not learned.get('usable', False)
        source = 'model_preliminary' if preliminary else 'model_validated'
        prefix = 'Evaluación preliminar: ' if preliminary else 'Evaluación del modelo: '
        title = prefix + ('posible ejecución incorrecta' if learned['error'] else 'parece correcta')
        decision_reason = calibration.get('reason', 'Falta una calibración verificada para dar un porcentaje y habilitar el verde.')
    elif measurement_ready:
        title = 'Sin desviaciones en los criterios medidos'
        source = 'measures'
        decision_reason = learned.get('reason', 'Sin detector disponible para calificar la repetición.')
    else:
        title = 'No concluyente: no hay datos suficientes para decidir'
        decision_reason = learned.get('reason', 'Sin detector disponible.')
    note = 'La evaluación no verifica toda la técnica ni predice lesiones.'
    if calibrated_ready or source == 'model_preliminary':
        note = 'El acierto con videos propios de cámara todavía necesita comprobarse. ' + note
    return {'title':title, 'exercise':exercise, 'rules':rules, 'learned':learned,
            'sample':sample, 'quality':quality, 'deviations':deviations,
            'measurement_ready':measurement_ready, 'prediction_ready':prediction_ready,
            'color':color, 'calibrated_ready':calibrated_ready,
            'decision_source':source, 'decision_reason':decision_reason, 'note':note}


def markdown_result(result, include_title=True):
    text = f"## {result['title']}\n\n" if include_title else ''
    if result.get('decision_reason'):
        text += result['decision_reason'] + '\n\n'
    if result.get('prediction_ready') and result['decision_source'] == 'measures':
        learned = result['learned']
        calibration = learned.get('calibration', {})
        if result.get('calibrated_ready'):
            text += f"**Modelo:** {calibration['p_correct']:.1%} de ejecución correcta (estimación calibrada de referencia)"
        else:
            text += '**Modelo:** ' + ('posible ejecución incorrecta' if learned['error'] else 'parece correcta')
            if not learned.get('usable'):text += ' (predicción preliminar)'
            if learned.get('near_threshold'):text += '; puntuación cerca del umbral'
        text += '.\n\n'
    text += '**Medidas observables en la vista seleccionada**\n\n'
    text += '| Criterio | Medida | Objetivo | Resultado |\n|---|---:|---:|---|\n'
    words = {'en_rango':'En rango','limite':'Cerca del límite',
             'fuera_de_rango':'Revisar','no_evaluable':'No disponible en esta grabación'}
    for check in result['rules']['checks']:
        value = '—' if check['status'] == 'no_evaluable' or check['value'] is None else str(check['value'])
        text += f"| {check['name']} | {value} | {check['direction']} {check['reference']} | {words[check['status']]} |\n"
    for check in result['deviations']:
        text += '\n**Sugerencia:** ' + check['advice'] + '\n'
    reasons = list(dict.fromkeys(c.get('reason','') for c in result['rules']['checks']
                                if c['status'] == 'no_evaluable' and c.get('reason')))
    for reason in reasons:
        text += '\n**Medida no disponible:** ' + reason + '\n'
    if not result['measurement_ready'] and result['rules']['reason'] and result['rules']['reason'] not in reasons:
        text += '\n**Recorrido 2D:** ' + result['rules']['reason'] + '\n'
    if result.get('quality'):
        text += '\nLado analizado: **' + result['quality']['side'] + '**.\n'
    return text + '\n' + result['note']
