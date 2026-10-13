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
        # El intervalo "limite" ya está dentro de la tolerancia de medida
        # configurada. No se admite cuando falta visibilidad o sólo se dispone
        # de una asimetría 3D que necesita confirmación independiente.
        if (rep['complete'] and rep['color']=='amarillo' and rep['reason']=='Hay un criterio cerca del límite.'
            and rep.get('visible_fraction',0)>=.8 and rep.get('measured_fraction',0)>=.8
            and rep.get('criterion_coverage',0)>=.75 and rep['checks'][0]['status'] in ('en_rango','limite')
            and not any(c['status']=='fuera_de_rango' or c.get('evidence_status')=='uncertain_3d' for c in rep['checks'])):
            rep.update(color='verde',reason='Los criterios revisados cumplen dentro de la tolerancia configurada.',tolerance_applied=True)
            measured=sum(c['weight'] for c in rep['checks'] if c['status']!='no_evaluable')
            passed=sum(c['weight'] for c in rep['checks'] if c['status'] in ('en_rango','limite'))
            rep['criterion_score']=passed/measured if measured else 0.
        failed=[c for c in rep['checks'] if c['status']=='fuera_de_rango']
        # Los fallos de una repetición completa y legible son evidencia técnica.
        # Una repetición cortada no demuestra por sí sola recorrido insuficiente.
        if rep.get('observable_error') or (rep['complete'] and rep['color']=='rojo'):
            for c in failed:
                consequence=c.get('consequence') or CONSEQUENCES.get(c['key'],'Revisa este patrón con un profesional antes de aumentar la carga.')
                name='Recorrido incompleto' if c['key']=='rango_incompleto' else 'Retorno incompleto' if c['key']=='extension' else c['name']
                errors.append(dict(key=c['key'],name=name,advice=c['advice'],consequence=consequence,
                                   repetition=rep['number'],side=c.get('side',rep['side']),source=c.get('source')))
    for observation in result.get('posture_observations',[]):
        for c in observation['checks']:
            if c['status']=='fuera_de_rango':
                errors.append(dict(key=c['key'],name=c['name'],advice=c['advice'],consequence=c['consequence'],
                                   repetition=None,side=c['side'],source=c.get('source'),
                                   start_time=observation['start_time'],end_time=observation['end_time'],scope='postura'))
    result['errors']=errors
    result['counts']={c:sum(r['color']==c for r in reps) for c in ['verde','rojo','amarillo']}
    complete=[r for r in reps if r['complete']]
    assessed=[r for r in complete if r.get('visible_fraction',0)>=.8 and r.get('measured_fraction',0)>=.8
              and r.get('criterion_coverage',0)>=.75]
    assessed_fraction=len(assessed)/len(complete) if complete else 0.
    confirmed_pass=bool(assessed and assessed_fraction>=.8 and all(r['color']=='verde' for r in assessed))
    readable=bool(complete and result['quality']['ready'])
    model=result.get('model',{'available':False});p=float(model.get('p_correct',.5))
    if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Probabilidad del modelo inválida.')
    if errors:
        result.update(color='rojo',status='incorrect',title='Ejercicio mal ejecutado',decision_source='error_observable',
                      reason='; '.join(dict.fromkeys(e['name'] for e in errors))+'.')
    elif readable and confirmed_pass and model.get('available') and .35<p<=.5:
        result.update(color='verde',status='correct',title='Ejercicio realizado correctamente',decision_source='criterios_y_modelo',
                      reason='Las repeticiones revisadas cumplen el recorrido y los criterios visibles, sin fallos confirmados.')
    elif readable and model.get('available') and p<=.5:
        result.update(color='rojo',status='incorrect',title='Ejercicio mal ejecutado',decision_source='modelo_binario',
                      reason='El modelo clasifica la ejecución como incorrecta. Las medidas visibles no permiten localizar un fallo específico.',
                      model_confidence=1-p)
    elif readable and model.get('available'):
        result.update(color='verde',status='correct',title='Ejercicio realizado correctamente',decision_source='modelo_rgb',
                      reason='La ejecución cumple la evaluación del video y no se detectaron errores claros.',model_confidence=p)
    elif complete and all(r['color']=='verde' for r in complete) and (not model.get('available') or p>=.5):
        result.update(color='verde',status='correct',title='Ejercicio realizado correctamente',decision_source='criterios_observables',
                      reason='Las repeticiones completas cumplen los criterios visibles. Los tramos cortados no se califican.')
    elif readable:
        result.update(color='amarillo',status='unreadable',title='No se pudo evaluar',decision_source='criterios_insuficientes',
                      reason='Se detectó el movimiento, pero faltan medidas fiables de uno o más criterios necesarios para calificarlo.')
    elif result.get('posture_observations'):
        result.update(color='amarillo',status='uncertain',title='No se pudo evaluar',decision_source='postura_parcial',
                      reason='Se pudo revisar la alineación en algunos tramos, sin un fallo confirmado. Falta una lectura continua del movimiento para calificar la ejecución completa.')
    else:
        result.update(color='amarillo',status='unreadable',title='No se pudo evaluar',decision_source='lectura_insuficiente',
                      reason='Falta una repetición completa o una lectura fiable de las articulaciones. Graba el cuerpo y los pies visibles desde el inicio hasta el retorno.')
        if result.get('reading',{}).get('ambiguous_fraction',0)>.2:
            result['reason']='Hay varias personas y no se pudo seguir a una con suficiente certeza. Activa Elegir persona en Ajustar ejercicio y toma.'
        elif result['exercise']=='squat' and result.get('reading',{}).get('detected_fraction',0)>.5:
            result['reason']='Se localizó a la persona, pero no se pudo medir rodillas, tobillos y pies con suficiente continuidad. No hay un fallo técnico confirmado con esas medidas.'
    result['feedback_version']=POLICY_VERSION
    result['app_revision']=APP_REVISION
    result['decision_thresholds']={'binary_p_error':.5,'confirmed_criteria_override_min_p_correct':.35,
                                   'confirmed_criteria_min_assessed_fraction':.8,
                                   'tie_break':'confirmed_measured_criteria_else_incorrect'}
    scores=[r['criterion_score'] for r in complete if r.get('criterion_score') is not None]
    result['criterion_score']=float(np.mean(scores)) if scores else None
    return result
