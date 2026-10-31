"""Validación anidada por persona, calibración separada y exportación portable."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from neural_models import fit,logits,probability,save_bundle

NEURAL=["gru","lstm","stgcn"]
CLASSIC=["random_forest","extra_trees","svm","logistic"]

def grouped(y,groups,count,seed):
    if len(set(groups))<count:raise ValueError(f"Se necesitan al menos {count} personas; hay {len(set(groups))}.")
    for attempt in range(30):
        folds=list(StratifiedGroupKFold(count,shuffle=True,random_state=seed+attempt).split(np.zeros(len(y)),y,groups))
        if all(len(set(y[a]))==2 and len(set(y[b]))==2 for a,b in folds):
            for a,b in folds:
                if set(groups[a])&set(groups[b]):raise AssertionError("Fuga de identidad.")
            return folds
    raise ValueError("No se pueden construir folds con ambas clases por persona. No se cambiará a división por video.")

def configurations(family,smoke=False):
    if family in NEURAL:
        return [{"hidden":h,"dropout":.2,"lr":1e-3} for h in ([32] if smoke else [32,64])]
    if family=="random_forest":return [{"n_estimators":160,"max_depth":6,"min_samples_leaf":3}]
    if family=="extra_trees":return [{"n_estimators":240,"max_depth":8,"min_samples_leaf":2}]
    return [{"C":v} for v in ([1] if smoke else [.1,1] if family=="logistic" else [1,10])]

def classic(family,config,seed):
    if family=="random_forest":return RandomForestClassifier(**config,class_weight="balanced",random_state=seed,n_jobs=2)
    if family=="extra_trees":return ExtraTreesClassifier(**config,class_weight="balanced",random_state=seed,n_jobs=2)
    if family=="svm":return make_pipeline(StandardScaler(),SVC(**config,gamma="scale",class_weight="balanced",random_state=seed))
    return make_pipeline(StandardScaler(),LogisticRegression(**config,class_weight="balanced",max_iter=2000,random_state=seed))

def score_classic(model,x):
    last=model.steps[-1][1] if hasattr(model,"steps") else model
    return model.predict_proba(x)[:,1] if hasattr(last,"estimators_") else model.decision_function(x)

def calibrate(score,y,seed):
    model=LogisticRegression(C=1,random_state=seed).fit(np.asarray(score)[:,None],y)
    slope=float(model.coef_[0,0]);intercept=float(model.intercept_[0])
    if slope<=0:
        # Calibración monótona conservadora para scores que no se separan.
        p=float(np.clip(np.mean(y),1e-5,1-1e-5));slope=0;intercept=float(np.log(p/(1-p)))
    return dict(coefficient=slope,intercept=intercept,monotonic=slope>0,n_calibration=len(y))

def selection(family,data,dev,inner_folds,config,seed,device):
    reports=[];best=None
    for params in configurations(family,config["mode"]!="tesis"):
        scores=[];epochs=[];histories=[]
        for k,(a,b) in enumerate(inner_folds):
            train,val=dev[a],dev[b]
            if family in NEURAL:
                model,norm,info=fit(family,params,data["x"],data["mask"],data["y"],train,val,seed+k,
                    config["epochs"],config["patience"],device)
                score=logits(model,norm.transform(data["x"][val],data["mask"][val]),device)
                epochs.append(info["best_epoch"]);histories.append(info)
            else:
                model=classic(family,params,seed+k);model.fit(data["stat"][train],data["y"][train]);score=score_classic(model,data["stat"][val])
            scores.append(float(roc_auc_score(data["y"][val],score)))
        report=dict(config=params,inner_auc=float(np.mean(scores)),inner_auc_sd=float(np.std(scores)),
            fold_auc=scores,epochs=epochs,histories=histories)
        reports.append(report)
        print(f"  {family} {params}: AUC interna {report['inner_auc']:.3f}",flush=True)
        if best is None or report["inner_auc"]>best["inner_auc"]:best=report
    return best,reports

def portable_classic(model):
    # Exportación sin pickle y sin importar el extractor MediaPipe.
    last=model.steps[-1][1] if hasattr(model,"steps") else model
    if hasattr(last,"estimators_"):
        trees=[]
        for member in last.estimators_:
            t=member.tree_;v=t.value[:,0,:]
            trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),
                threshold=t.threshold.tolist(),p_error=(v[:,1]/np.maximum(v.sum(1),1e-9)).tolist()))
        return dict(kind="forest",trees=trees)
    scaler=model.steps[0][1];base=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist())
    if isinstance(last,LogisticRegression):return dict(base,kind="linear",coefficients=last.coef_[0].tolist(),intercept=float(last.intercept_[0]))
    return dict(base,kind="svm",support_vectors=last.support_vectors_.tolist(),dual_coefficients=last.dual_coef_[0].tolist(),
        gamma=float(last._gamma),intercept=float(last.intercept_[0]))

def fit_final(family,best,data,dev,cal,seed,config,device,folder,metadata):
    started=time.perf_counter();params=best["config"]
    if family in NEURAL:
        epochs=max(1,int(np.median(best["epochs"])))
        model,norm,info=fit(family,params,data["x"],data["mask"],data["y"],dev,None,seed,
            config["epochs"],config["patience"],device,fixed_epochs=epochs)
        raw=logits(model,norm.transform(data["x"][cal],data["mask"][cal]),device)
        calibration=calibrate(raw,data["y"][cal],seed)
        metadata=dict(metadata,fit_info=info,approved=False)
        save_bundle(folder,model,family,params,norm,calibration,metadata)
        # Referencia SHAP exclusivamente de personas de entrenamiento.
        reference=np.zeros((64,33,2),np.float32)
        for node in [11,12,13,14,23,24,25,26]:
            for c in [6,7]:
                for t in range(64):
                    observed=data["x"][dev,t,node,c][data["mask"][dev,t,node,c]]
                    if len(observed):reference[t,node,c-6]=np.median(observed)
        np.savez_compressed(Path(folder)/"shap_reference.npz",angles=reference)
        def predict(indices):return probability(logits(model,norm.transform(data["x"][indices],data["mask"][indices]),device),calibration)
        count=info["parameters"]
    else:
        model=classic(family,params,seed);model.fit(data["stat"][dev],data["y"][dev]);raw=score_classic(model,data["stat"][cal])
        calibration=calibrate(raw,data["y"][cal],seed)
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        (folder/"classical.json").write_text(json.dumps(dict(model=portable_classic(model),calibration=calibration,**metadata)))
        def predict(indices):return probability(score_classic(model,data["stat"][indices]),calibration)
        count=None
    return predict,dict(fit_seconds=time.perf_counter()-started,parameters=count,calibration=calibration)

def load_data(path,excluded):
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k].copy() for k in ["x","mask","stat"]};records=json.loads(str(z["metadata"].item()))
    keep=np.array([r["subject"] not in excluded for r in records])
    data={k:v[keep] for k,v in data.items()};data["records"]=[r for r,k in zip(records,keep) if k]
    data["y"]=np.array([r["error"] for r in data["records"]],int)
    data["groups"]=np.array([r["subject"] for r in data["records"]])
    return data

def run(data_root,run_root,config_path):
    data_root=Path(data_root).resolve();run_root=Path(run_root).resolve();run_root.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(config_path).read_text());source=json.loads((data_root/"data.json").read_text())
    from neural_features import fingerprint
    if source["representation"]!=fingerprint():raise ValueError("Cambió la representación: repetir preparación antes de entrenar.")
    digest=hashlib.sha256(json.dumps(dict(config=config,data=source),sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()
    experiment=run_root/digest[:16];experiment.mkdir(parents=True,exist_ok=True)
    (experiment/"config.json").write_text(json.dumps(config,indent=2))
    (experiment/"data.json").write_text(json.dumps(source,ensure_ascii=False,indent=2))
    device="cuda" if torch.cuda.is_available() and config.get("gpu",True) else "cpu"
    environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,device=device,
        gpu=torch.cuda.get_device_name(0) if device=="cuda" else None,seed=config["seed"],fingerprint=digest)
    (experiment/"environment.json").write_text(json.dumps(environment,indent=2))
    (experiment/"requirements_lock.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    from research_reports import summarize,approve
    oof=[];plans=[];final=[]
    for exercise in config["exercises"]:
        file=data_root/f"{exercise}.npz"
        if not file.exists():raise ValueError(f"No hay datos elegibles para {exercise}; revisar auditoría.")
        data=load_data(file,config["excluded_subjects"]);y,g=data["y"],data["groups"];n=len(y)
        if len(set(g))<max(config["outer_folds"]+3,10):raise ValueError(f"{exercise}: personas insuficientes para protocolo anidado y calibración separada.")
        outer=grouped(y,g,config["outer_folds"],config["seed"])
        for fold,(train,test) in enumerate(outer):
            local_dev,local_cal=grouped(y[train],g[train],config["calibration_folds"],config["seed"]+fold+91)[0]
            dev,cal=train[local_dev],train[local_cal]
            inner=grouped(y[dev],g[dev],config["inner_folds"],config["seed"]+fold+201)
            for role,indices in [("development",dev),("calibration",cal),("test",test)]:
                plans.extend(dict(exercise=exercise,fold=fold,role=role,id=data["records"][i]["id"],subject=g[i],error=int(y[i])) for i in indices)
            bests={};predictions={}
            for family in CLASSIC+NEURAL:
                folder=experiment/exercise/f"fold_{fold}"/family;checkpoint=folder/"completed.json"
                if checkpoint.exists():report=json.loads(checkpoint.read_text());best=report["best"];pred=np.array(report["predictions"])
                else:
                    print(f"{exercise}: fold {fold+1}/{len(outer)}, {family}",flush=True)
                    best,candidates=selection(family,data,dev,inner,config,config["seed"]+fold,device)
                    meta=dict(exercise=exercise,extractor_id=source["extractor_id"],pipeline_signature=source["pipeline_signature"],
                        train_ids=[data["records"][i]["id"] for i in dev],calibration_ids=[data["records"][i]["id"] for i in cal],
                        test_ids=[data["records"][i]["id"] for i in test],scope="validación interna por persona; etiqueta por video",mode=config["mode"])
                    predict,info=fit_final(family,best,data,dev,cal,config["seed"]+fold,config,device,folder,meta)
                    started=time.perf_counter();pred=predict(test);info["ms_per_video_tensor"]=(time.perf_counter()-started)*1000/len(test)
                    report=dict(best=best,candidates=candidates,predictions=pred.tolist(),**info)
                    checkpoint.write_text(json.dumps(report,ensure_ascii=False,indent=2))
                bests[family]=best;predictions[family]=pred
                for i,p in zip(test,pred):oof.append(dict(data["records"][i],fold=fold,family=family,p_error=float(p),**{k:report[k] for k in ["fit_seconds","parameters","ms_per_video_tensor"]}))
            for name,families in [("seleccion_neuronal",NEURAL),("seleccion_clasica",CLASSIC)]:
                winner=max(families,key=lambda f:bests[f]["inner_auc"])
                for i,p in zip(test,predictions[winner]):oof.append(dict(data["records"][i],fold=fold,family=name,chosen=winner,p_error=float(p)))
            pd.DataFrame(oof).to_csv(experiment/"predicciones_oof.csv",index=False)
            pd.DataFrame(plans).to_csv(experiment/"particiones.csv",index=False)
        # Selección final por folds internos. Los tests externos nunca eligen familia.
        final_folder=experiment/exercise/"final";done=final_folder/"completed.json"
        if done.exists():final_record=json.loads(done.read_text())
        else:
            dev,cal=grouped(y,g,config["calibration_folds"],config["seed"]+900)[0]
            inner=grouped(y[dev],g[dev],config["inner_folds"],config["seed"]+901)
            choices={}
            for family in NEURAL:
                best,candidates=selection(family,data,dev,inner,config,config["seed"],device);choices[family]=best
            winner=max(NEURAL,key=lambda f:choices[f]["inner_auc"])
            _,info=fit_final(winner,choices[winner],data,dev,cal,config["seed"],config,device,final_folder,
                dict(exercise=exercise,extractor_id=source["extractor_id"],pipeline_signature=source["pipeline_signature"],
                    train_ids=[data["records"][i]["id"] for i in dev],calibration_ids=[data["records"][i]["id"] for i in cal],
                    test_ids=[],scope="modelo final para piloto; sin cohorte externa independiente",mode=config["mode"]))
            final_record=dict(exercise=exercise,family=winner,choices=choices,**info)
            done.write_text(json.dumps(final_record,indent=2))
        final.append(final_record)
    frame=pd.DataFrame(oof);frame.to_csv(experiment/"predicciones_oof.csv",index=False)
    pd.DataFrame(plans).to_csv(experiment/"particiones.csv",index=False)
    (experiment/"seleccion_final.json").write_text(json.dumps(final,indent=2))
    summarize(experiment,data_root,config);approve(experiment,config)
    (run_root/"latest.json").write_text(json.dumps(dict(path=str(experiment),fingerprint=digest,complete=True)))
    print("EXPERIMENTO_COMPLETO",experiment,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--runs",required=True);p.add_argument("--config",required=True)
    a=p.parse_args();run(a.data,a.runs,a.config)
