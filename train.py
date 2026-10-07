"""Alternativa rápida: RandomForest de ángulos + CV anidada por persona + calibración.
Las anotaciones binarias y los subtipos revisados son tareas separadas.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from geometry import features,feature_names,SCHEMA,EXERCISES,TAGS
from runtime_model import forest_probability

CODE_VERSION='private_v4_source_calibration'


def metrics(y,p,t):
    pred=np.asarray(p)>=np.asarray(t);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'samples':len(y),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'sensitivity':float(tp/(tp+fn)) if tp+fn else None,
            'specificity':float(tn/(tn+fp)) if tn+fp else None,
            'f1_error':float(f1_score(y,pred,zero_division=0)),
            'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]]}


def threshold(y,p):
    # Peso de sensibilidad; especificidad mínima explícita, solo datos de entrenamiento internos.
    candidates=[]
    for t in np.linspace(.05,.95,91):
        m=metrics(y,p,t)
        if m['specificity'] is not None and m['sensitivity'] is not None:
            score=(2*m['sensitivity']+m['specificity'])/3
            candidates.append((m['specificity']>=.60,score,m['balanced_accuracy'],-abs(t-.5),float(t)))
    return max(candidates)[-1] if candidates else .5


def splits(x,y,groups,k=5):
    ng=len(np.unique(groups))
    # Cada partición requiere ambas clases en train y prueba para comparar los umbrales.
    for count in range(min(k,ng),1,-1):
        candidate=list(StratifiedGroupKFold(count,shuffle=True,random_state=42).split(x,y,groups))
        if all(len(np.unique(y[a]))==2 and len(np.unique(y[b]))==2 for a,b in candidate):return candidate
    raise ValueError('Faltan personas con ambas clases para validar sin mezclar sujetos.')


def weights(groups,sources):
    # Cada fuente y persona tiene el mismo peso total; no sobre-muestreamos test.
    result=np.empty(len(groups))
    for source in np.unique(sources):
        mask=sources==source;people=np.unique(groups[mask])
        for person in people:
            idx=mask&(groups==person);result[idx]=1/(len(people)*idx.sum())
    return result/result.mean()


def fit(x,y,groups,sources):
    model=RandomForestClassifier(n_estimators=160,max_depth=6,min_samples_leaf=3,
         max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=-1)
    model.fit(x,y,sample_weight=weights(groups,sources));return model


def serialize(model):
    trees=[]
    for est in model.estimators_:
        t=est.tree_;value=t.value[:,0,:];p=value[:,1]/np.maximum(value.sum(axis=1),1e-9)
        trees.append({'children_left':t.children_left.tolist(),'children_right':t.children_right.tolist(),
           'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'p_error':p.tolist()})
    return trees


def digest(items,exercise):
    h=hashlib.sha256((CODE_VERSION+SCHEMA+exercise).encode())
    for item in sorted(items,key=lambda a:a['id']):
        h.update(json.dumps({k:item[k] for k in ['id','subject','source','error','tags','metadata']},sort_keys=True).encode())
        h.update(item['coords'].tobytes())
    return h.hexdigest()


def good(m):
    return m['balanced_accuracy']>=.7 and (m['sensitivity'] or 0)>=.65 and (m['specificity'] or 0)>=.65


def train_exercise(items,exercise,out_root):
    out=Path(out_root)/exercise;out.mkdir(parents=True,exist_ok=True)
    fingerprint=digest(items,exercise);p=out/'modelo.json'
    if p.exists():
        old=json.loads(p.read_text())
        if old.get('fingerprint')==fingerprint:
            print(exercise,': sin cambios; se reutiliza el modelo.');return old
    x=np.stack([features(item['coords'],exercise,item['metadata'].get('target')) for item in items]);y=np.array([a['error'] for a in items])
    groups=np.array([a['subject'] for a in items]);sources=np.array([a['source'] for a in items])
    outer=splits(x,y,groups);probs=np.full(len(y),np.nan);cuts=np.zeros(len(y));folds=[]
    # El fingerprint evita mezclar checkpoints de datos o configuración distintos.
    work=out/'checkpoints'/fingerprint[:16];work.mkdir(parents=True,exist_ok=True)
    for num,(tr,te) in enumerate(outer):
        checkpoint=work/f'fold_{num}.npz'
        if checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as f:
                if not np.array_equal(te,f['test_index']):raise ValueError('Checkpoint con partición distinta.')
                probs[te]=f['prob'];cuts[te]=f['threshold']
        else:
            xt,yt,gt,st=x[tr],y[tr],groups[tr],sources[tr]
            calibration=np.full(len(tr),np.nan)
            for inner_tr,inner_val in splits(xt,yt,gt,k=3):
                m=fit(xt[inner_tr],yt[inner_tr],gt[inner_tr],st[inner_tr])
                calibration[inner_val]=m.predict_proba(xt[inner_val])[:,1]
            cut=threshold(yt,calibration)
            source_cuts={}
            for source in np.unique(st):
                mask=st==source
                if len(np.unique(yt[mask]))==2 and len(np.unique(gt[mask]))>=3:
                    source_cuts[source]=threshold(yt[mask],calibration[mask])
            m=fit(xt,yt,gt,st);probs[te]=m.predict_proba(x[te])[:,1]
            cuts[te]=[source_cuts.get(source,cut) for source in sources[te]]
            np.savez_compressed(checkpoint,test_index=te,prob=probs[te],threshold=cuts[te])
        fm=metrics(y[te],probs[te],cuts[te]);fm['test_subjects']=groups[te].tolist()
        folds.append(fm);print(exercise,'fold',num+1,'/',len(outer),'balanced=',round(fm['balanced_accuracy'],3),flush=True)
    overall=metrics(y,probs,cuts);by_source={str(s):metrics(y[sources==s],probs[sources==s],cuts[sources==s]) for s in np.unique(sources)}
    # Final: umbral de OOF de todos los datos y ajuste de pesos sobre todos. No se promedian cortes de otras redes.
    final_cut=threshold(y,probs)
    final_source_cuts={str(source):threshold(y[sources==source],probs[sources==source])
                       for source in np.unique(sources) if len(np.unique(y[sources==source]))==2}
    model=fit(x,y,groups,sources);trees=serialize(model)
    camera=sources=='camera';camera_people=len(np.unique(groups[camera]))
    camera_ok=bool(camera_people>=5 and np.sum(y[camera]==0)>=20 and np.sum(y[camera]==1)>=20 and good(by_source['camera']))
    profiles=[]
    for view,target in sorted(set((str(a['metadata'].get('view')),float(a['metadata'].get('target',0))) for a in items if a['source']=='camera')):
        mask=np.array([a['source']=='camera' and str(a['metadata'].get('view'))==view and float(a['metadata'].get('target',0))==target for a in items])
        score=metrics(y[mask],probs[mask],cuts[mask]);people=len(np.unique(groups[mask]))
        passed=people>=5 and np.sum(y[mask]==0)>=20 and np.sum(y[mask]==1)>=20 and good(score)
        profiles.append({'view':view,'target':target,'subjects':people,'metrics':score,'passed':bool(passed)})
    metadata={'version':datetime.now(timezone.utc).isoformat(),'fingerprint':fingerprint,
        'feature_schema':SCHEMA,'feature_names':feature_names(exercise),'trees':trees,
        'threshold':final_cut,'threshold_by_source':final_source_cuts,'metrics':overall,'folds':folds,'source_metrics':by_source,
        'fold_summary':{'balanced_accuracy_mean':float(np.mean([a['balanced_accuracy'] for a in folds])),
                        'balanced_accuracy_std':float(np.std([a['balanced_accuracy'] for a in folds]))},
        'feature_low':(np.min(x,axis=0)-.3*np.ptp(x,axis=0)-1e-3).tolist(),
        'feature_high':(np.max(x,axis=0)+.3*np.ptp(x,axis=0)+1e-3).tolist(),
        'accepted':good(overall),'camera_validated':camera_ok,'camera_subjects':camera_people,'camera_profiles':profiles,
        'evaluation':'StratifiedGroupKFold anidado, no LOSO; umbrales por fuente sin usar prueba',
        'subtypes':'Sin etiquetas públicas de subtipo: desconocido. Revisiones humanas se conservan separadas.'}
    # Comprobar la exportación numérica contra sklearn antes de instalarla.
    np.testing.assert_allclose([forest_probability(metadata,a) for a in x[:12]],model.predict_proba(x[:12])[:,1],atol=1e-7)
    if p.exists():
        history=out/'history';history.mkdir(exist_ok=True)
        (history/f'{json.loads(p.read_text())["fingerprint"][:16]}.json').write_bytes(p.read_bytes())
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(metadata,ensure_ascii=False));os.replace(tmp,p)
    np.savez_compressed(out/'predicciones_oof.npz',y=y,prob=probs,threshold=cuts,subject=groups,source=sources)
    # Separación explícita: nunca redistribuir una clase vacía ni evaluar reglas como ground truth.
    audit={'binary':{'correct':int(np.sum(y==0)),'incorrect':int(np.sum(y==1))},
           'subjects':len(np.unique(groups)),'subtypes':{}}
    for tag in TAGS[exercise]:
        reviewed=[a for a in items if a['tags'] is not None]
        positives=[a for a in reviewed if tag in a['tags']]
        audit['subtypes'][tag]={'reviewed_positive_samples':len(positives),
            'positive_subjects':len(set(a['subject'] for a in positives)),
            'unknown_samples':sum(a['tags'] is None or (a['error'] and not a['tags']) for a in items)}
    (out/'auditoria.json').write_text(json.dumps(audit,indent=2))
    return metadata


def train_all(items,out_root):
    results={}
    for exercise in EXERCISES:
        selected=[a for a in items if a['exercise']==exercise]
        if len(selected)<20 or len(set(a['subject'] for a in selected))<5:
            print(exercise,': requiere >=20 muestras y >=5 personas. Se mantienen las medidas de cámara.');continue
        try:results[exercise]=train_exercise(selected,exercise,out_root)
        except ValueError as e:print(exercise,':',e)
        from subtypes import train_reviewed_subtypes
        train_reviewed_subtypes(selected,exercise,out_root)
    return results


if __name__=='__main__':
    from datasets import download_rehab,load_rehab,load_legacy_cache,complete_irds
    from feedback import load_reviews
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='datos_publicos/rehab246')
    parser.add_argument('--legacy-cache');parser.add_argument('--output',default=os.environ.get('MOVEAI_MODELS_DIR','modelos_rapidos'))
    args=parser.parse_args();items=load_rehab(download_rehab(args.data))
    if args.legacy_cache:items+=load_legacy_cache(args.legacy_cache)
    items=complete_irds(items,Path(args.data).parent/'irds')
    items+=load_reviews();train_all(items,args.output)
