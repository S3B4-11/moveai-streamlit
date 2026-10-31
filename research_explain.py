"""SHAP agrupado sobre ocho curvas angulares; evaluación fuera de muestra."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neural_features import ANGLES
from neural_models import load_bundle,logits,probability

def explain(root,data_root,count=4,nsamples=256):
    root=Path(root);dest=root/"shap";dest.mkdir(exist_ok=True)
    config=json.loads((root/"config.json").read_text());oof=pd.read_csv(root/"predicciones_oof.csv");rows=[];checks=[]
    rng=np.random.default_rng(config["seed"])
    for exercise,ex in oof[oof.family=="seleccion_neuronal"].groupby("exercise"):
        with np.load(Path(data_root)/f"{exercise}.npz",allow_pickle=False) as z:
            x=z["x"];mask=z["mask"];records=json.loads(str(z["metadata"].item()))
        by_id={r["id"]:i for i,r in enumerate(records)}
        # Ambas clases si existen, selección fija sin elegir por confianza.
        chosen=[]
        for _,f in ex.groupby("error"):
            chosen.extend(f.iloc[rng.permutation(len(f))[:max(1,count//2)]].to_dict("records"))
        for record in chosen[:count]:
            folder=root/exercise/f"fold_{record['fold']}"/record["chosen"]
            model,norm,meta=load_bundle(folder)
            if record["id"] in meta["train_ids"] or record["id"] in meta["calibration_ids"]:raise AssertionError("SHAP debe explicar un video externo al modelo del fold.")
            with np.load(folder/"shap_reference.npz",allow_pickle=False) as ref:background=ref["angles"]
            index=by_id[record["id"]];original=x[index].copy();missing=mask[index]
            def evaluate_coalitions(coalitions):
                tensors=np.repeat(original[None],len(coalitions),axis=0)
                for group,(node,key,_) in enumerate(ANGLES):
                    for c in [6,7]:
                        tensors[:,:,node,c]=coalitions[:,group,None]*original[:,node,c]+(1-coalitions[:,group,None])*background[:,node,c-6]
                masks=np.repeat(missing[None],len(coalitions),axis=0)
                tensors[~masks]=0
                return probability(logits(model,norm.transform(tensors,masks)),meta["calibration"])
            explainer=shap.KernelExplainer(evaluate_coalitions,np.zeros((1,8)),link="identity")
            values=np.asarray(explainer.shap_values(np.ones((1,8)),nsamples=nsamples,l1_reg=0,silent=True)).reshape(-1)
            baseline=float(np.asarray(explainer.expected_value).item());p=float(evaluate_coalitions(np.ones((1,8)))[0])
            checks.append(dict(id=record["id"],exercise=exercise,fold=record["fold"],family=record["chosen"],p_error=p,
                oof_p_error=record["p_error"],baseline=baseline,sum_shap=float(values.sum()),additivity_error=float(p-baseline-values.sum()),
                baseline_scope="curvas medianas de entrenamiento; coordenadas y máscaras del video fijas"))
            for (_,key,_),v in zip(ANGLES,values):rows.append(dict(id=record["id"],exercise=exercise,family=record["chosen"],angle=key,shap=float(v),error=record["error"]))
            plt.figure(figsize=(6,3.8));plt.barh([a[1] for a in ANGLES],values,color=["#d1495b" if v>0 else "#16a99a" for v in values])
            plt.xlabel("Contribución a P(bad) respecto de la referencia angular");plt.title(f"{exercise} · {record['id'][:8]}");plt.tight_layout()
            plt.savefig(dest/f"local_{exercise}_{record['id'][:12]}.png",dpi=200,bbox_inches="tight");plt.close()
            print("SHAP",exercise,record["id"][:12],"P(bad)",round(p,4),flush=True)
    pd.DataFrame(rows).to_csv(dest/"atribuciones_angulares.csv",index=False);pd.DataFrame(checks).to_csv(dest/"auditoria_aditividad.csv",index=False)
    note="SHAP agrupa ocho curvas angulares: los canales 2D y 3D de cada articulación se reemplazan juntos. Referencia: mediana por instante sólo del entrenamiento del fold. Coordenadas, visibilidad y máscaras del video permanecen fijas. Explicación condicional de canales redundantes; las perturbaciones pueden no corresponder a poses físicamente realizables. No atribuye causalidad, lesiones ni etiquetas de error a repeticiones. Modelos explicados: selección neuronal fuera de muestra."
    (dest/"interpretacion.txt").write_text(note)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--data",required=True)
    p.add_argument("--count",type=int,default=4);p.add_argument("--nsamples",type=int,default=256)
    a=p.parse_args();explain(a.root,a.data,a.count,a.nsamples)
