"""Evaluación privada con medición observable y detector aprendido independiente."""
import os
import numpy as np
from pathlib import Path
from geometry import EXERCISES, rule_assessment
from pose import extract_video
from runtime_model import predict_model,predict_subtypes


def evaluate(path,exercise,view='perfil',side='derecha',target=None):
    if exercise not in EXERCISES:raise ValueError('Ejercicio desconocido.')
    sample=extract_video(path)
    sample.update(view=view,side=side,target=target)
    sample['trainable']=bool(sample['coverage']>=.8 and sample['multiple_people']==0)
    rules=rule_assessment(sample['image'],sample['visibility'],exercise,view,side,target)
    sample['target']=rules['target']
    measurement_ready=bool(sample['trainable'] and rules['evaluable'])
    joints=[11,12,23,24,25,26,27,28] if exercise in ('squat','inline_lunge') else [11,12,13,14,15,16,23,24]
    feature_visible=float((np.min(sample['visibility'][:,joints],axis=1)>=.55).mean())>=.8
    sample['trainable']=bool(measurement_ready and feature_visible)
    if sample['multiple_people']:
        title='Graba con una sola persona en cuadro';sample['trainable']=False
    elif sample['coverage']<.8:
        title='Falta visibilidad durante la repetición';sample['trainable']=False
    elif not rules['evaluable']:title='No se pudo medir el recorrido'
    else:title='Sin desviaciones en los criterios medidos'
    learned=predict_model(sample['world'],exercise,view=view,target=rules['target'])
    learned['reviewed_subtypes']=predict_subtypes(sample['world'],exercise,target=rules['target'])
    deviations=[c for c in rules['checks'] if c['status']=='fuera_de_rango']
    if measurement_ready and deviations:title='Revisa: '+', '.join(c['name'].lower() for c in deviations)
    elif measurement_ready and learned.get('usable') and learned['error']:
        title='El patrón necesita revisión: no se identificó un error específico'
    if not measurement_ready:
        for c in rules['checks']:c['status']='no_evaluable'
    return {'title':title,'exercise':exercise,'rules':rules,'learned':learned,
            'sample':sample,'deviations':deviations if measurement_ready else [],'measurement_ready':measurement_ready,
            'note':'El resultado cubre los criterios medidos en esta vista; no verifica toda la técnica ni predice lesiones.'}


def markdown_result(r):
    text=f"## {r['title']}\n\n"
    text+='| Criterio | Medida | Objetivo | Resultado |\n|---|---:|---:|---|\n'
    words={'en_rango':'En rango','limite':'Cerca del límite','fuera_de_rango':'Revisar','no_evaluable':'Sin medir'}
    for c in r['rules']['checks']:
        val='—' if c['status']=='no_evaluable' or c['value'] is None else str(c['value'])
        text+=f"| {c['name']} | {val} | {c['direction']} {c['reference']} | {words[c['status']]} |\n"
    for c in r['deviations']:text+='\n**Sugerencia:** '+c['advice']+'\n'
    missing=sorted({c['view'] for c in r['rules']['checks'] if c['status']=='no_evaluable'})
    if missing:text+='\nPara completar las medidas, vuelve a grabar de '+ ' / '.join(missing)+'.\n'
    if not r['measurement_ready']:text+='\n'+r['rules']['reason']+'\n'
    return text+'\n'+r['note']
