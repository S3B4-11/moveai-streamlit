"""RGB → pose → detector portable, validación y calibración por persona."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,recall_score,precision_score,brier_score_loss,confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from camera_model import FEATURE_NAMES,features,signature,raw_score,fuse,hard_failure
from feedback import review_rows
from geometry import SCHEMA,EXERCISES
from pose import extract_video,extractor_id

DEVELOPMENT_SUBJECTS={"mendeley_rgb:1"}

def cached_sample(row,cache,extractor):
    if "sample" in row:return row["sample"]
    path=Path(cache)/"poses"/(row["id"]+"_"+extractor[:16]+".npz");path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            sample={k:f[k].copy() for k in ["world","image","visibility","times","detected","ambiguous"]}
            sample.update(json.loads(str(f["metadata"].item())),previews=[])
        return sample
    sample=extract_video(row["path"],previews=False)
    meta={k:sample[k] for k in ["coverage","duration","pose_version","extractor_id","pipeline"]}
    tmp=path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,**{k:sample[k] for k in ["world","image","visibility","times","detected","ambiguous"]},metadata=json.dumps(meta))
    tmp.replace(path);return sample

def prepare_one(row,cache,extractor,version,poses_only=False):
    os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
    with threadpool_limits(limits=1):
        if poses_only:cached_sample(row,cache,extractor);return dict(id=row["id"],cached=True)
        fp=Path(cache)/"features"/(row["id"]+"_"+version[:16]+".json");fp.parent.mkdir(parents=True,exist_ok=True)
        if fp.exists():return json.loads(fp.read_text())
        x,r=features(cached_sample(row,cache,extractor),row["exercise"])
        record={k:row[k] for k in ["id","exercise","error","subject","source","view"]}
        record.update(x=x.tolist(),criteria_color=r["color"],model_ready=r["quality"]["ready"],
                      hard_failure=hard_failure(r),repetitions=len(r["repetitions"]),
                      complete_repetitions=sum(v["complete"] for v in r["repetitions"]))
        fp.write_text(json.dumps(record));return record

def load_rows(manifest,reviews,extractor):
    rows=json.loads(Path(manifest).read_text())["rows"] if manifest else []
    imported={}
    for archive in reviews:
        for row in review_rows(archive,extractor):
            previous=imported.get(row["id"])
            if previous and (previous["subject"]!=row["subject"] or previous["exercise"]!=row["exercise"]):
                raise ValueError("Una revisión duplicada tiene otra persona o ejercicio.")
            if previous is None or row["reviewed_at"]>previous["reviewed_at"]:imported[row["id"]]=row
    for row in rows:
        if row["id"] in imported:raise ValueError("Un video del dataset fue incluido como si fuera otra persona; elimina esa revisión.")
    return rows+list(imported.values())

def candidates(seed):
    return {
      "random_forest":RandomForestClassifier(n_estimators=160,max_depth=6,min_samples_leaf=3,class_weight="balanced",random_state=seed,n_jobs=2),
      "extra_trees":ExtraTreesClassifier(n_estimators=240,max_depth=8,min_samples_leaf=2,class_weight="balanced",random_state=seed,n_jobs=2),
      "svm_c1":make_pipeline(StandardScaler(),SVC(C=1,gamma="scale",class_weight="balanced",random_state=seed)),
      "svm_c10":make_pipeline(StandardScaler(),SVC(C=10,gamma="scale",class_weight="balanced",random_state=seed)),
      "logistic_c01":make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",max_iter=2000,random_state=seed)),
      "logistic_c1":make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight="balanced",max_iter=2000,random_state=seed))}

def model_score(estimator,x):
    last=estimator.steps[-1][1] if hasattr(estimator,"steps") else estimator
    return estimator.predict_proba(x)[:,1] if hasattr(last,"estimators_") else estimator.decision_function(x)

def group_folds(x,y,groups,count=3,seed=12):
    splitter=StratifiedGroupKFold(n_splits=min(count,len(set(groups))),shuffle=True,random_state=seed)
    folds=list(splitter.split(x,y,groups))
    for train,test in folds:
        if set(groups[train])&set(groups[test]):raise AssertionError("Se mezcló una persona entre entrenamiento y prueba.")
        if len(set(y[train]))<2:raise ValueError("Un fold no tiene ambas etiquetas. Añade personas con ambas clases.")
    return folds

def calibrated_ensemble(x,y,groups,seed=12):
    repeats=[group_folds(x,y,groups,3,seed+101*r) for r in range(3)];best=None
    for name in candidates(seed):
        oof=np.zeros((3,len(y)));members=[]
        for r,folds in enumerate(repeats):
            for f,(train,test) in enumerate(folds):
                estimator=candidates(seed+101*r)[name];estimator.fit(x[train],y[train]);oof[r,test]=model_score(estimator,x[test])
                if f==r:members.append(estimator)
        # Se calibra la media de tres scores fuera de muestra, el mismo tipo
        # de media que se usa en la inferencia, en vez de promediar sigmoides.
        oof=oof.mean(axis=0)
        auc=roc_auc_score(y,oof)
        if best is None or auc>best[0]:best=(auc,name,oof,members)
    auc,name,oof,members=best
    sigmoid=LogisticRegression(C=1e6,solver="lbfgs",random_state=seed);sigmoid.fit(oof[:,None],y)
    coefficient=float(sigmoid.coef_[0,0]);intercept=float(sigmoid.intercept_[0])
    if coefficient<=0:raise ValueError("El detector no aprendió una separación estable en los folds internos.")
    def prediction(new_x):
        score=np.mean([model_score(m,new_x) for m in members],axis=0)
        return sigmoid.predict_proba(score[:,None])[:,1]
    return name,members,dict(coefficient=coefficient,intercept=intercept),prediction

def portable(estimator):
    if hasattr(estimator,"steps"):
        scaler=estimator.steps[0][1];svm=estimator.steps[1][1]
        if isinstance(svm,LogisticRegression):
            return dict(kind="linear",mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                        coefficients=svm.coef_[0].tolist(),intercept=float(svm.intercept_[0]))
        return dict(kind="svm",mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),support_vectors=svm.support_vectors_.tolist(),
                    dual_coefficients=svm.dual_coef_[0].tolist(),gamma=float(svm._gamma),intercept=float(svm.intercept_[0]))
    trees=[]
    for estimator_tree in estimator.estimators_:
        tree=estimator_tree.tree_;v=tree.value[:,0,:]
        trees.append(dict(left=tree.children_left.tolist(),right=tree.children_right.tolist(),feature=tree.feature.tolist(),
                          threshold=tree.threshold.tolist(),p_error=(v[:,1]/np.maximum(v.sum(axis=1),1e-9)).tolist()))
    return dict(kind="forest",trees=trees)

def limits(x):
    low,high=x.min(axis=0),x.max(axis=0);margin=.3*np.maximum(high-low,1.)
    return low-margin,high+margin

def metrics(y,p,groups):
    pred=(p>=.5).astype(int);green=p<=.25;red=p>=.75
    return dict(n=len(y),subjects=len(set(groups)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                sensitivity_error=float(recall_score(y,pred,pos_label=1,zero_division=0)),
                specificity=float(recall_score(y,pred,pos_label=0,zero_division=0)),auc_roc=float(roc_auc_score(y,p)),
                brier=float(brier_score_loss(y,p)),confusion_correct_error=confusion_matrix(y,pred,labels=[0,1]).tolist(),
                green_precision=float((y[green]==0).mean()) if green.any() else None,green_n=int(green.sum()),
                green_subjects=len(set(groups[green])),red_precision=float((y[red]==1).mean()) if red.any() else None,red_n=int(red.sum()))

def color_metrics(y,colors,groups):
    colors=np.asarray(colors);green=colors=="verde";red=colors=="rojo";definite=green|red
    return dict(green_n=int(green.sum()),green_subjects=len(set(groups[green])),red_n=int(red.sum()),yellow_n=int((~definite).sum()),
                green_precision=float((y[green]==0).mean()) if green.any() else None,
                red_precision=float((y[red]==1).mean()) if red.any() else None,
                definite_coverage=float(definite.mean()),accuracy_when_definite=float(((colors[definite]=="rojo")==y[definite]).mean()) if definite.any() else None)

def train_exercise(records,exercise,output,extractor,version):
    # Coincide con la puerta de calidad de inferencia. Las poses incoherentes
    # son fallos de lectura, no ejemplos de técnica correcta/incorrecta.
    rows=[r for r in records if r["exercise"]==exercise and r["model_ready"]
          and r["complete_repetitions"]>0 and r["x"][FEATURE_NAMES.index("reading_fraction")]>=.35]
    rejected=sum(r["exercise"]==exercise for r in records)-len(rows)
    if len(rows)<20 or len(set(r["subject"] for r in rows))<6 or len(set(r["error"] for r in rows))<2:
        print(exercise,"sin suficientes videos/personas con ambas clases; se conservan criterios observables.",flush=True);return None
    x=np.asarray([r["x"] for r in rows],np.float64);y=np.asarray([r["error"] for r in rows]);groups=np.asarray([r["subject"] for r in rows])
    keep=np.asarray([s not in DEVELOPMENT_SUBJECTS for s in groups]);xx,yy,gg=x[keep],y[keep],groups[keep]
    if len(set(gg))<6:raise ValueError("No hay suficientes personas fuera del grupo de desarrollo.")
    rr=[r for r,k in zip(rows,keep) if k];oof=np.zeros(len(yy));eligible=np.zeros(len(yy),bool);fold_metrics=[]
    with threadpool_limits(limits=2):
        for index,(train,test) in enumerate(group_folds(xx,yy,gg,5,42),1):
            name,members,calibration,prediction=calibrated_ensemble(xx[train],yy[train],gg[train],100+index)
            p=prediction(xx[test]);oof[test]=p;low,high=limits(xx[train]);eligible[test]=np.mean((xx[test]<low)|(xx[test]>high),axis=1)<=.25
            m=metrics(yy[test],p,gg[test]);m.update(fold=index,candidate=name,test_subjects=sorted(set(gg[test])))
            fold_metrics.append(m);print(f"{exercise}: fold {index}/5 · {name} · balanced_accuracy={m['balanced_accuracy']:.3f} · AUC={m['auc_roc']:.3f}",flush=True)
        m=metrics(yy,oof,gg)
        colors=[fuse(r["criteria_color"],1-p,bool(r["model_ready"] and e),r["hard_failure"])[0] for r,p,e in zip(rr,oof,eligible)]
        application=color_metrics(yy,colors,gg);criteria=color_metrics(yy,[r["criteria_color"] for r in rr],gg)
        approved=bool(m["balanced_accuracy"]>=.7 and m["sensitivity_error"]>=.65 and m["specificity"]>=.65
                      and application["green_n"]>=15 and application["green_subjects"]>=10
                      and (application["green_precision"] or 0)>=.75)
        m.update(application=application,criteria_only=criteria,folds=fold_metrics,rejected_unreadable=rejected,
                 development_subjects_excluded=sorted(DEVELOPMENT_SUBJECTS),approved=approved,
                 approval_rule="Piloto privado: balanced_accuracy>=0.70; sensibilidad/especificidad>=0.65; precision verde>=0.75 en >=15 videos y >=10 personas; las vistas de una persona no son muestras independientes")
        m["by_view"]={view:metrics(yy[mask],oof[mask],gg[mask]) for view in sorted(set(r["view"] for r in rr))
                      if len(set(yy[mask:=np.asarray([r["view"]==view for r in rr])]))==2}
        name,members,calibration,prediction=calibrated_ensemble(x,y,groups,2026)
    model=dict(kind="ensemble",members=[portable(member) for member in members],score_calibration=calibration,
               schema=SCHEMA,input_domain="rgb_video",signature=version,extractor_id=extractor,feature_names=FEATURE_NAMES,
               exercise=exercise,approved=approved,candidate=name,metrics=m)
    low,high=limits(x);model.update(feature_low=low.tolist(),feature_high=high.tolist())
    expected=prediction(x[:10]);observed=np.asarray([raw_score(model,v) for v in x[:10]])
    np.testing.assert_allclose(expected,observed,atol=2e-6,rtol=2e-6)
    dest=Path(output)/exercise;dest.mkdir(parents=True,exist_ok=True)
    (dest/"modelo.json").write_text(json.dumps(model,separators=(",",":")))
    (dest/"metricas.json").write_text(json.dumps(m,ensure_ascii=False,indent=2))
    np.savez_compressed(dest/"evaluacion_rgb.npz",y=yy,p_error=oof,subjects=gg,views=np.asarray([r["view"] for r in rr]),
                        colors=np.asarray(colors),ids=np.asarray([r["id"] for r in rr]))
    print(exercise,"RGB:",json.dumps({k:m[k] for k in ["n","subjects","balanced_accuracy","sensitivity_error","specificity","approved"]}),flush=True)
    return m

def run(args):
    extractor=extractor_id();version=signature();rows=load_rows(args.manifest,args.reviews,extractor)
    if not rows:raise ValueError("No hay datos. Ejecuta rgb_dataset.py o importa revisiones humanas.")
    records=[];failures=[]
    workers=max(1,min(args.workers,4));print("Videos:",len(rows),"· procesos:",workers,"· pose:",extractor[:12],flush=True)
    # Sólo se retienen vectores; ni frames ni coordenadas quedan en los futuros.
    if args.cached_only:
        for row in rows:
            p=Path(args.cache)/"features"/(row["id"]+"_"+version[:16]+".json")
            if not p.exists():raise ValueError("Faltan características preparadas: "+row["id"])
            records.append(json.loads(p.read_text()))
    else:
        extract_records(rows,args,extractor,version,workers,records,failures)
    Path(args.cache).mkdir(parents=True,exist_ok=True)
    (Path(args.cache)/"rechazados.json").write_text(json.dumps(failures,ensure_ascii=False,indent=2))
    if args.poses_only:return
    records.sort(key=lambda r:(r["exercise"],r["subject"],r["id"]))
    summary={}
    for exercise in EXERCISES:
        m=train_exercise(records,exercise,args.output,extractor,version)
        if m is not None:summary[exercise]=m
    Path(args.output).mkdir(parents=True,exist_ok=True)
    (Path(args.output)/"resumen_rgb.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))

def extract_records(rows,args,extractor,version,workers,records,failures):
    # MediaPipe/JAX mantienen hilos nativos. Spawn evita heredarlos por fork y
    # quedar esperando el cierre del pool cuando las poses ya se escribieron.
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        pending={pool.submit(prepare_one,row,args.cache,extractor,version,args.poses_only):row["id"] for row in rows}
        total=len(pending);completed=0
        for future in as_completed(pending):
            sid=pending[future];completed+=1
            try:records.append(future.result())
            except Exception as error:failures.append(dict(id=sid,error=str(error)));print("Omitido:",sid[:10],str(error),flush=True)
            pending[future]=None
            if completed%10==0 or completed==total:print(f"Pose/características: {completed}/{total}",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--manifest");parser.add_argument("--reviews",nargs="*",default=[])
    parser.add_argument("--cache",default="rgb_cache");parser.add_argument("--output",default="camera_modelos")
    parser.add_argument("--workers",type=int,default=2);parser.add_argument("--poses-only",action="store_true")
    parser.add_argument("--cached-only",action="store_true")
    run(parser.parse_args())
