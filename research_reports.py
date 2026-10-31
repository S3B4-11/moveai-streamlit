"""Tablas y figuras científicas construidas únicamente con predicciones reales."""
import argparse
import html
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,precision_score,recall_score,
    roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix,roc_curve,precision_recall_curve)

def metrics(frame):
    y=frame.error.to_numpy(int);p=frame.p_error.to_numpy(float);pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    green=p<=.25;red=p>=.75
    return dict(videos=len(y),persons=frame.subject.nunique(),accuracy=accuracy_score(y,pred),
        balanced_accuracy=balanced_accuracy_score(y,pred),f1_macro=f1_score(y,pred,average="macro",zero_division=0),
        precision_error=precision_score(y,pred,zero_division=0),sensitivity_error=recall_score(y,pred,zero_division=0),
        specificity_good=tn/max(tn+fp,1),auc=roc_auc_score(y,p) if len(set(y))==2 else None,
        average_precision=average_precision_score(y,p),brier=brier_score_loss(y,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        confident_coverage=float((green|red).mean()),uncertain_fraction=float((~(green|red)).mean()),
        green_precision=float((y[green]==0).mean()) if green.any() else None,
        red_precision=float((y[red]==1).mean()) if red.any() else None)

def color(row):
    if not row.get("model_ready",False):return row["criteria_color"]
    p=1-row["p_error"]
    if p>=.75:return "amarillo" if row.get("hard_failure",False) else "verde"
    if p<=.25:return "amarillo" if row["criteria_color"]=="verde" else "rojo"
    if row["criteria_color"]=="verde" and p>=.5:return "verde"
    return "amarillo"

def fused(frame):
    colors=frame.apply(lambda r:color(r.to_dict()),axis=1);green=colors=="verde";red=colors=="rojo"
    return dict(fused_green=int(green.sum()),fused_green_subjects=int(frame.loc[green,"subject"].nunique()),
        fused_green_precision=float((frame.loc[green,"error"]==0).mean()) if green.any() else None,
        fused_red=int(red.sum()),fused_red_precision=float((frame.loc[red,"error"]==1).mean()) if red.any() else None,
        fused_yellow=int((colors=="amarillo").sum()))

def bootstrap(frame,replicates=2000,seed=42):
    rng=np.random.default_rng(seed);subjects=frame.subject.unique();values=[]
    chunks={s:frame[frame.subject==s] for s in subjects}
    for _ in range(replicates):
        draw=pd.concat([chunks[s] for s in rng.choice(subjects,len(subjects),replace=True)],ignore_index=True)
        if draw.error.nunique()<2:continue
        values.append(balanced_accuracy_score(draw.error,draw.p_error>=.5))
    return np.percentile(values,[2.5,97.5]).tolist() if values else [None,None]

def paired_bootstrap(left,right,replicates=2000,seed=42):
    merge=left[["id","subject","error","p_error"]].merge(right[["id","p_error"]],on="id",suffixes=("_nn","_ml"),validate="one_to_one")
    if len(merge)!=len(left) or len(merge)!=len(right):raise ValueError("Comparación no pareada.")
    rng=np.random.default_rng(seed);subjects=merge.subject.unique();chunks={s:merge[merge.subject==s] for s in subjects};delta=[]
    for _ in range(replicates):
        draw=pd.concat([chunks[s] for s in rng.choice(subjects,len(subjects),replace=True)])
        if draw.error.nunique()<2:continue
        delta.append(balanced_accuracy_score(draw.error,draw.p_error_nn>=.5)-balanced_accuracy_score(draw.error,draw.p_error_ml>=.5))
    return dict(delta_ba=balanced_accuracy_score(merge.error,merge.p_error_nn>=.5)-balanced_accuracy_score(merge.error,merge.p_error_ml>=.5),
        ci_low=float(np.percentile(delta,2.5)),ci_high=float(np.percentile(delta,97.5)),participants=len(subjects))

def savefig(path):
    plt.tight_layout();plt.savefig(path.with_suffix(".png"),dpi=200,bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"),bbox_inches="tight");plt.close()

def summarize(root,data_root,config):
    root=Path(root);figs=root/"figuras";figs.mkdir(exist_ok=True)
    frame=pd.read_csv(root/"predicciones_oof.csv");table=[];views=[];folds=[];comparisons=[];calibration=[]
    reps=config.get("bootstrap",2000)
    for (exercise,family),f in frame.groupby(["exercise","family"]):
        low,high=bootstrap(f,reps,config["seed"])
        row=dict(exercise=exercise,family=family,**metrics(f),ba_ci_low=low,ba_ci_high=high,
            **(fused(f) if exercise in ["squat","elbow_flexion"] else {}))
        table.append(row)
        for view,v in f.groupby("view"):views.append(dict(exercise=exercise,family=family,view=view,**metrics(v)))
        for fold,v in f.groupby("fold"):folds.append(dict(exercise=exercise,family=family,fold=fold,**metrics(v)))
        plt.figure(figsize=(3.8,3.4));cm=confusion_matrix(f.error,f.p_error>=.5,labels=[0,1]);plt.imshow(cm,cmap="Blues")
        for i in range(2):
            for j in range(2):plt.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=16)
        plt.xticks([0,1],["good","bad"]);plt.yticks([0,1],["good","bad"]);plt.xlabel("Predicción");plt.ylabel("Etiqueta de video")
        plt.title(f"{exercise} · {family}",fontsize=9);savefig(figs/f"cm_{exercise}_{family}")
        bins=np.minimum((f.p_error.to_numpy()*10).astype(int),9)
        for b in range(10):
            local=f[bins==b]
            if len(local):calibration.append(dict(exercise=exercise,family=family,bin=b,videos=len(local),mean_probability=local.p_error.mean(),observed_error=local.error.mean()))
    result=pd.DataFrame(table);result.to_csv(root/"metricas_modelos.csv",index=False)
    pd.DataFrame(views).to_csv(root/"metricas_por_vista.csv",index=False);pd.DataFrame(folds).to_csv(root/"metricas_por_fold.csv",index=False)
    pd.DataFrame(calibration).to_csv(root/"calibracion.csv",index=False)
    for exercise,ex in frame.groupby("exercise"):
        comparison=paired_bootstrap(ex[ex.family=="seleccion_neuronal"],ex[ex.family=="seleccion_clasica"],reps,config["seed"])
        comparisons.append(dict(exercise=exercise,**comparison))
        plt.figure(figsize=(6,4))
        for family,f in ex.groupby("family"):
            a,b,_=roc_curve(f.error,f.p_error);plt.plot(a,b,label=f"{family}: {roc_auc_score(f.error,f.p_error):.3f}")
        plt.plot([0,1],[0,1],"k--",alpha=.3);plt.xlabel("Falsos positivos (good → bad)");plt.ylabel("Sensibilidad a bad")
        plt.legend(fontsize=7);plt.title(f"ROC fuera de muestra · {exercise}");savefig(figs/f"roc_{exercise}")
        plt.figure(figsize=(6,4))
        for family,f in ex.groupby("family"):
            p,r,_=precision_recall_curve(f.error,f.p_error);plt.plot(r,p,label=family)
        plt.xlabel("Sensibilidad a bad");plt.ylabel("Precisión de bad");plt.legend(fontsize=7);savefig(figs/f"pr_{exercise}")
        plt.figure(figsize=(6,4));c=pd.DataFrame(calibration)
        for family,f in c[c.exercise==exercise].groupby("family"):plt.plot(f.mean_probability,f.observed_error,".-",label=family)
        plt.plot([0,1],[0,1],"k--",alpha=.3);plt.xlabel("Probabilidad calibrada de bad");plt.ylabel("Fracción observada de bad");plt.legend(fontsize=7)
        savefig(figs/f"reliability_{exercise}")
    pd.DataFrame(comparisons).to_csv(root/"comparacion_pareada.csv",index=False)
    history=[]
    for file in root.glob("*/fold_*/*/completed.json"):
        report=json.loads(file.read_text())
        for h,info in enumerate(report["best"].get("histories",[])):
            for row in info["history"]:history.append(dict(exercise=file.parents[2].name,fold=file.parents[1].name,family=file.parent.name,inner_fold=h,**row))
    curves=pd.DataFrame(history);curves.to_csv(root/"curvas_aprendizaje.csv",index=False)
    if len(curves):
        for (ex,family),f in curves.groupby(["exercise","family"]):
            plt.figure(figsize=(5,3.5))
            for _,h in f.groupby(["fold","inner_fold"]):plt.plot(h.epoch,h.train_loss,alpha=.3)
            plt.xlabel("Época");plt.ylabel("BCE ponderada de entrenamiento");plt.title(f"{ex} · {family}");savefig(figs/f"loss_{ex}_{family}")
    for name in ["auditoria_dataset.csv","cinematica_repeticiones.csv","distribucion_dataset.csv"]:shutil.copy2(Path(data_root)/name,root/name)
    lines=["# Resultados MoveAI generados desde la ejecución",f"Modo: {config['mode']}. Semilla: {config['seed']}.",
        "La clase positiva es bad (1). Cada fila de prueba representa un video; sus repeticiones no tienen etiquetas individuales.",
        "Los modelos se evalúan con las mismas personas y videos, en validación anidada agrupada por sujeto. La calibración usa sujetos reservados dentro de entrenamiento.",
        "GRU/LSTM/ST-GCN usan el mismo tensor 64×33×10. RF/Extra Trees/SVM/logística usan 184 variables RGB 4 en curl y sentadilla; en flexión y press usan 1056 resúmenes de los mismos landmarks. No es una ablación que aísle exclusivamente la arquitectura.",
        "La selección_neuronal y seleccion_clasica eligen la familia únicamente por AUC interna de cada fold. Sus OOF evalúan el procedimiento de selección, no una cohorte externa del modelo final.",
        "IC 95%: bootstrap por participante. La comparación pareada conserva todos los videos/vistas de cada participante. No mide la variación de volver a entrenar con semillas diferentes.",
        "La latencia guardada corresponde a inferencia sobre tensores ya extraídos; no equivale a FPS de video ni incluye MediaPipe. Las redes pueden usar GPU mientras los baselines usan CPU.",
        "Las coordenadas world y ángulos son estimaciones monoculares, sin validación frente a MoCap. Los umbrales de interfaz son heurísticos y no estiman lesiones, fuerza articular ni diagnósticos.",
        "RGB 4 conserva su página y sus reglas por repetición. Los dos ejercicios nuevos se evalúan en investigación; no se añadieron opciones a su interfaz.","", "| Ejercicio | Familia | N videos | N personas | BA | AUC | F1 macro | Brier |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:lines.append(f"| {r['exercise']} | {r['family']} | {r['videos']} | {r['persons']} | {r['balanced_accuracy']:.3f} | {r['auc']:.3f} | {r['f1_macro']:.3f} | {r['brier']:.3f} |")
    lines.extend(["","## Comparación de procedimientos seleccionados"])
    for r in comparisons:lines.append(f"{r['exercise']}: ΔBA neural−clásico = {r['delta_ba']:.3f}; IC95% [{r['ci_low']:.3f}, {r['ci_high']:.3f}].")
    lines.extend(["","## Datos para redactar la tesis", "Usar config.json, data.json y particiones.csv para metodología; metricas_modelos.csv y figuras para resultados; cinemática para ejemplos descriptivos, sin afirmar exactitud por repetición.",
        "La aprobación de piloto es un filtro operativo basado en validación interna y no certifica generalización clínica. Conservar los archivos de esta ejecución y el notebook con sus salidas."])
    text="\n\n".join(lines);(root/"resultados_para_tesis.md").write_text(text)
    html_content="<html lang='es'><meta charset='utf-8'><title>MoveAI · Resultados</title><style>body{font:16px sans-serif;max-width:1100px;margin:36px auto}table{border-collapse:collapse;font-size:12px}td,th{padding:5px;border:1px solid #ddd}img{max-width:700px}pre{white-space:pre-wrap}</style><h1>MoveAI · Evidencia de ejecución</h1>"
    html_content+=f"<pre>{html.escape(text)}</pre>"+result.to_html(index=False,float_format=lambda x:f"{x:.3f}")
    for p in sorted(figs.glob("roc_*.png")):html_content+=f"<img src='figuras/{p.name}'>"
    (root/"reporte_tesis.html").write_text(html_content+"</html>")

def approve(root,config):
    root=Path(root);table=pd.read_csv(root/"metricas_modelos.csv");records=[]
    for exercise in config["exercises"]:
        row=table[(table.exercise==exercise)&(table.family=="seleccion_neuronal")].iloc[0].to_dict()
        classic=table[(table.exercise==exercise)&(table.family=="seleccion_clasica")].iloc[0]
        data_meta=json.loads((root/"data.json").read_text())
        checks=dict(full_run=config["mode"]=="tesis" and data_meta.get("limit_per_exercise",1)==0,balanced_accuracy=row["balanced_accuracy"]>=.7,
            sensitivity=row["sensitivity_error"]>=.65,specificity=row["specificity_good"]>=.65,
            at_least_classic=row["balanced_accuracy"]>=classic.balanced_accuracy,
            available_in_original_page=exercise in ["elbow_flexion","squat"],
            enough_green=row.get("fused_green",0)>=15,green_subjects=row.get("fused_green_subjects",0)>=10,
            green_precision=row.get("fused_green_precision",0)>=.75)
        path=root/exercise/"final"/"model.json";meta=json.loads(path.read_text())
        checks["monotonic_calibration"]=meta["calibration"]["coefficient"]>0
        checks={k:bool(v) for k,v in checks.items()}
        approved=all(checks.values());meta.update(approved=approved,metrics=row,approval_checks=checks,
            scope="Piloto RGB con validación interna por persona; probabilidad global por video. No valida errores por repetición ni riesgo de lesión.")
        path.write_text(json.dumps(meta,ensure_ascii=False,indent=2));records.append(dict(exercise=exercise,family=meta["family"],approved=approved,checks=checks))
    (root/"aprobacion_piloto.json").write_text(json.dumps(records,ensure_ascii=False,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--data",required=True);p.add_argument("--config",required=True)
    a=p.parse_args();config=json.loads(Path(a.config).read_text());summarize(a.root,a.data,config);approve(a.root,config)
