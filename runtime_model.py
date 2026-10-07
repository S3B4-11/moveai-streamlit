"""Inferencia portable en JSON, sin pickle ni dependencias de TensorFlow."""
from pathlib import Path
import json
import os
import numpy as np
from geometry import features, SCHEMA


def forest_probability(model,x):
    x=np.asarray(x,dtype=np.float32);out=[]
    for tree in model['trees']:
        node=0
        while tree['children_left'][node]!=-1:
            feature=tree['feature'][node]
            node=tree['children_left'][node] if x[feature]<=tree['threshold'][node] else tree['children_right'][node]
        out.append(tree['p_error'][node])
    return float(np.mean(out))


def predict_model(coords,exercise,root=None,view=None,target=None):
    root=Path(root or os.environ.get('MOVEAI_MODELS_DIR',Path(__file__).parent/'modelos_rapidos'));p=root/exercise/'modelo.json'
    if not p.exists():return {'available':False,'reason':'Sin detector entrenado para este ejercicio.'}
    m=json.loads(p.read_text())
    if m['feature_schema']!=SCHEMA:raise ValueError('Modelo incompatible con la extracción actual.')
    x=features(coords,exercise,target);prob=forest_probability(m,x)
    low=np.asarray(m['feature_low']);high=np.asarray(m['feature_high'])
    out_of_domain=float(np.mean((x<low)|(x>high)))>.25
    # Evaluar el sensor/cámara propio exige datos de cámara etiquetados de varias personas.
    camera_validated=bool(m.get('camera_validated',False) and any(p['passed'] and p['view']==view and abs(p['target']-float(target or 0))<1 for p in m.get('camera_profiles',[])))
    usable=bool(m.get('accepted',False) and camera_validated and not out_of_domain)
    return {'available':True,'usable':usable,'p_error':prob,'error':prob>=m.get('threshold_by_source',{}).get('camera',m['threshold']),
            'out_of_domain':out_of_domain,'camera_validated':camera_validated,
            'metrics':m['metrics'],'model_version':m['version'],'reason':
            'Detector validado con videos propios.' if usable else
            'El detector se mantiene como comparación hasta validarlo con videos propios de varias personas.'}


def predict_subtypes(coords,exercise,root=None,target=None):
    root=Path(root or os.environ.get('MOVEAI_MODELS_DIR',Path(__file__).parent/'modelos_rapidos'))
    p=root/exercise/'subtipos_revisados.json'
    if not p.exists():return {'suggestions':[],'reason':'Sin etiquetas de subtipo revisadas suficientes.'}
    m=json.loads(p.read_text())
    if m['schema']!=SCHEMA:return {'suggestions':[],'reason':'Versión de características diferente.'}
    x=features(coords,exercise,target);out=[]
    for tag,head in m['heads'].items():
        if head['accepted']:
            prob=forest_probability(head,x)
            if prob>=head['threshold']:out.append({'tag':tag,'score':prob})
    return {'suggestions':out,'coverage':m['status']}
