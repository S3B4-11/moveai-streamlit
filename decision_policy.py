"""Separa fallo técnico, incertidumbre y falta real de lectura del video."""
import numpy as np
from feedback_presentation import APP_REVISION, POLICY_VERSION

CONSEQUENCES={
 'compensacion_tronco':'El balanceo puede redistribuir el esfuerzo hacia el tronco y la zona lumbar. Revisa el control del movimiento y la carga.',
 'movimiento_brazo':'El impulso con el brazo puede cambiar el esfuerzo y el control del ejercicio. Conviene revisar la carga y el movimiento.',
 'asimetria':'Una participación desigual puede repartir la carga de forma desigual entre las piernas.',
 'extension':'Un retorno distinto al objetivo puede cambiar el recorrido y el estímulo del ejercicio. No demuestra una lesión.',
 'rango_incompleto':'Un recorrido menor al objetivo cambia el estímulo del ejercicio. No significa por sí solo que haya una lesión.',
 'codo_flexionado':'Esta variante cambia el recorrido y el esfuerzo respecto al objetivo seleccionado.'}

def apply_policy(result):
    reps=result['repetitions'];errors=[]
    for rep in reps:
        for c in rep['checks']:
            if c['key']=='asimetria' and c['status']=='fuera_de_rango' and rep['angle_source']=='3D estimada':
                c.update(status='limite',evidence_status='uncertain_3d',advice='La diferencia entre piernas es una estimación de profundidad. Confírmala con otra toma antes de considerarla un fallo.')
        if rep['color']=='rojo' and not rep.get('observable_error') and not any(c['status']=='fuera_de_rango' for c in rep['checks']):
            rep.update(color='amarillo',reason='La diferencia estimada entre piernas requiere confirmación.')
        failed=[c for c in rep['checks'] if c['status']=='fuera_de_rango']
        # Los fallos de una repetición completa y legible son evidencia técnica.
        # Una repetición cortada no demuestra por sí sola recorrido insuficiente.
        if rep.get('observable_error') or (rep['complete'] and rep['color']=='rojo'):
            for c in failed:
                consequence=c.get('consequence') or CONSEQUENCES.get(c['key'],'Revisa este patrón con un profesional antes de aumentar la carga.')
                errors.append(dict(key=c['key'],name=c['name'],advice=c['advice'],consequence=consequence,
                                   repetition=rep['number'],side=c.get('side',rep['side']),source=c.get('source')))
    result['errors']=errors
    result['counts']={c:sum(r['color']==c for r in reps) for c in ['verde','rojo','amarillo']}
    complete=[r for r in reps if r['complete']]
    readable=bool(complete and result['quality']['ready'])
    model=result.get('model',{'available':False});p=float(model.get('p_correct',.5))
    if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Probabilidad del modelo inválida.')
    if errors:
        result.update(color='rojo',status='incorrect',title='Necesita corrección',decision_source='error_observable',
                      reason='; '.join(dict.fromkeys(e['name'] for e in errors))+'.')
    elif readable and model.get('available') and p<=.35:
        result.update(color='rojo',status='incorrect',title='Necesita corrección',decision_source='modelo_binario',
                      reason='El modelo clasifica la ejecución como incorrecta. Las medidas visibles no permiten localizar un fallo específico.',
                      model_confidence=1-p)
    elif readable and model.get('available') and p>=.75:
        result.update(color='verde',status='correct',title='Ejecución correcta',decision_source='modelo_rgb',
                      reason='La ejecución cumple la evaluación del video y no se detectaron errores claros.',model_confidence=p)
    elif complete and all(r['color']=='verde' for r in complete) and (not model.get('available') or p>=.5):
        result.update(color='verde',status='correct',title='Ejecución correcta',decision_source='criterios_observables',
                      reason='Las repeticiones completas cumplen los criterios visibles. Los tramos cortados no se califican.')
    elif readable:
        result.update(color='amarillo',status='uncertain',title='Resultado no concluyente',decision_source='resultado_incierto',
                      reason='Hay datos del movimiento, pero no una conclusión clara sobre la técnica. No se considera un error confirmado.')
        if complete and all(r['color']=='verde' for r in complete):
            result['reason']='Los criterios medidos no muestran fallos, pero el modelo no permite confirmar la ejecución del video.'
    else:
        result.update(color='amarillo',status='unreadable',title='No se pudo evaluar',decision_source='lectura_insuficiente',
                      reason='Falta una repetición completa o una lectura fiable de las articulaciones. Graba el cuerpo y los pies visibles desde el inicio hasta el retorno.')
    result['feedback_version']=POLICY_VERSION
    result['app_revision']=APP_REVISION
    result['decision_thresholds']={'correct':.75,'incorrect':.65}
    return result
