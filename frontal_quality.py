"""No compara flexión 3D de rodilla con una proyección frontal sin profundidad.

La reparación sólo elimina ese motivo concreto, manteniendo las comprobaciones
de visibilidad, longitud de segmentos y máscaras. Los tensores entrenados no cambian.
"""
from pathlib import Path
import json
import numpy as np

PREFIX='Las estimaciones del recorrido difieren mucho'

def repair_frontal_projection(result):
    if result['exercise']!='squat':return False
    repaired=set()
    for track in result['tracks']:
        if (track['view_observed']=='frente' and track['source']=='3D estimada'
            and track['reason'].startswith(PREFIX) and track['pose_visibility']>=.8
            and np.mean(track['valid'])>=.8 and track['bone_variation']<=.25):
            track['reason']='';repaired.add(track['side'])
    if not repaired:return False
    for rep in result['repetitions']:
        if (rep['side'] not in repaired or not rep['complete'] or not rep['reason'].startswith(PREFIX)
            or rep['visible_fraction']<.8 or rep['measured_fraction']<.8 or rep['criterion_coverage']<.75):continue
        failed=[c for c in rep['checks'] if c['status']=='fuera_de_rango']
        if failed:rep.update(color='rojo',reason='Revisar: '+', '.join(c['name'] for c in failed)+'.')
        elif any(c['status']=='limite' for c in rep['checks']):rep.update(color='amarillo',reason='Hay un criterio cerca del límite.')
        elif rep['checks'][0]['status']=='en_rango' and rep['criterion_score']>=.75:
            rep.update(color='verde',reason='Los criterios medidos de la repetición se cumplen.')
    result['quality']['ready']=bool(any(r['complete'] for r in result['repetitions']) and all(t['pose_visibility']>=.8 and not t['reason'] for t in result['tracks']))
    result['sample']['trainable']=result['quality']['ready']
    result['frontal_projection_repaired']=True
    return True

def predict_resolved_frontal(sample,exercise,result,fallback):
    if not result.get('frontal_projection_repaired') or not result['quality']['ready']:return fallback
    from camera_model import features,FEATURE_NAMES,SCHEMA,signature,raw_score
    path=Path(__file__).parent/'camera_modelos'/exercise/'modelo.json'
    if not path.exists():return fallback
    model=json.loads(path.read_text())
    if (not model.get('approved') or model.get('schema')!=SCHEMA or model.get('input_domain')!='rgb_video'
        or model.get('signature')!=signature() or model.get('extractor_id')!=sample.get('extractor_id')
        or model.get('feature_names')!=FEATURE_NAMES):return fallback
    x,_=features(sample,exercise)
    if np.mean((x<np.asarray(model['feature_low']))|(x>np.asarray(model['feature_high'])))>.25:return fallback
    p=float(raw_score(model,x))
    return dict(available=True,p_error=p,p_correct=1-p,metrics=model['metrics'],
                scope='Detector RGB; comparación 2D/3D frontal corregida sin cambiar sus variables.',quality_adapter='frontal_projection')
