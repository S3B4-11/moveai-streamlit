"""Presenta por separado lo medido en una repetición y el juicio del video."""
APP_REVISION = '2026.10.13-r8'
POLICY_VERSION = 'moveai_feedback_v8'

def repetition_label(rep, result):
    if rep['color'] == 'rojo':
        return 'A corregir'
    if rep['color'] == 'verde':
        return 'Correcta' if result['color'] == 'verde' else 'Sin fallos medidos'
    return 'Sin evaluar'

def repetition_color(rep, result):
    if rep['color'] == 'verde' and result['color'] != 'verde':
        return 'neutro'
    return rep['color']

def count_labels(result):
    return ('Correctas' if result['color'] == 'verde' else 'Sin fallos medidos',
            'A corregir', 'Sin evaluar / por confirmar')
