"""Valida cada repetición sin confundir portadas/cierres con oclusión interna.

Adaptador de evaluación: no modifica poses, variables ni firmas del clasificador.
El modelo conserva su filtro original de calidad del video completo.
"""
import numpy as np
from analysis import assess
from motion import track

OCCLUSION = 'La extremidad activa queda oculta en parte del video. Grábala visible durante todo el recorrido.'

def repair_repetition_quality(result):
    sample=result['sample'];repaired=[]
    for index,rep in enumerate(result['repetitions']):
        if not rep['complete'] or rep['reason']!=OCCLUSION or rep.get('components'):continue
        a,b=rep['start'],rep['end']
        local=dict(sample)
        for key in ('world','image','visibility','times','detected','ambiguous'):
            if key in sample:local[key]=np.asarray(sample[key])[a:b]
        if len(local['times'])<8:continue
        checked=track(local,result['exercise'],rep['side'])
        original=next((t for t in result['tracks'] if t['side']==rep['side']),None)
        if (original is None or checked['reason'] or checked['pose_visibility']<.8
            or checked['visibility']<.8 or checked['source']!=original['source']
            or checked['view_observed']!=original['view_observed']):continue
        # Conserva las medidas originales; sólo retira el veto global tras
        # verificar de nuevo visibilidad, segmentos y proyección en este ciclo.
        safe=dict(original,reason='')
        boundaries={k:rep[k] for k in ('start','peak','end','complete','reason')}
        boundaries['reason']=''
        revised=assess(sample,result['exercise'],safe,boundaries,result['target'])
        revised.update(number=rep['number'],quality_scope='repetition',global_quality_reason=OCCLUSION)
        result['repetitions'][index]=revised;repaired.append(rep['number'])
    if repaired:
        result['repetition_quality_repair']={'repetitions':repaired,'scope':'Sólo repeticiones completas y visibles; los tramos cortados no se califican.'}
    return result
