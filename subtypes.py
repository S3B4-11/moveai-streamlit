"""Etiquetas de error revisadas: cabezas binarias independientes, sin inventar clases vacías."""
import json
from pathlib import Path
import numpy as np
from geometry import features,TAGS,SCHEMA
from train import splits,fit,metrics,threshold,serialize


def train_reviewed_subtypes(items,exercise,out_root):
    reviewed=[a for a in items if a['source']=='camera' and a['tags'] is not None
              and (not a['error'] or a['tags'])]
    status={};heads={}
    for tag in TAGS[exercise]:
        positive=[a for a in reviewed if tag in a['tags']]
        npos=len(positive);nneg=len(reviewed)-npos
        people=len(set(a['subject'] for a in reviewed))
        pos_people=len(set(a['subject'] for a in positive))
        status[tag]={'positive':npos,'negative':nneg,'subjects':people,'positive_subjects':pos_people,'trained':False}
        if npos<10 or nneg<10 or people<5 or pos_people<3:continue
        x=np.stack([features(a['coords'],exercise,a['metadata'].get('target')) for a in reviewed])
        y=np.array([int(tag in a['tags']) for a in reviewed]);g=np.array([a['subject'] for a in reviewed]);src=np.repeat('camera',len(y))
        try:outer=splits(x,y,g)
        except ValueError:continue
        prob=np.zeros(len(y));cuts=np.zeros(len(y))
        for tr,te in outer:
            try:inner=splits(x[tr],y[tr],g[tr],3)
            except ValueError:break
            calibrate=np.zeros(len(tr))
            for a,b in inner:calibrate[b]=fit(x[tr][a],y[tr][a],g[tr][a],src[tr][a]).predict_proba(x[tr][b])[:,1]
            cut=threshold(y[tr],calibrate)
            m=fit(x[tr],y[tr],g[tr],src[tr]);prob[te]=m.predict_proba(x[te])[:,1];cuts[te]=cut
        else:
            score=metrics(y,prob,cuts);status[tag]['metrics']=score
            accept=score['f1_error']>=.6 and score['sensitivity']>=.65 and score['specificity']>=.65
            status[tag]['trained']=True;status[tag]['accepted']=accept
            final=fit(x,y,g,src)
            heads[tag]={'trees':serialize(final),'threshold':threshold(y,prob),'metrics':score,'accepted':accept}
    out=Path(out_root)/exercise;out.mkdir(parents=True,exist_ok=True)
    (out/'subtipos_revisados.json').write_text(json.dumps({'schema':SCHEMA,'status':status,'heads':heads},ensure_ascii=False))
    return status
