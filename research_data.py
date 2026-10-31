"""Auditoría, extracción RGB y datos por ejercicio. Sin etiquetas por repetición."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
from geometry import signals,smooth,q
from motion import repetitions
from neural_features import encode,fingerprint,ANGLES

NAMES={"elbow_flexion":"Curl de bíceps","squat":"Sentadilla","push_up":"Flexión de brazos","shoulder_press":"Press de hombros"}

def kinematics(sample,exercise):
    activity="knee" if exercise=="squat" else "arm" if exercise=="shoulder_press" else "elbow"
    channels={node:(key,joints) for node,key,joints in ANGLES}
    nodes=[25,26] if activity=="knee" else [11,12] if activity=="arm" else [13,14]
    sig=signals(sample["world"]);tracks=[];times=sample["times"]
    for node in nodes:
        key,joints=channels[node]
        valid=(sample["visibility"][:,joints]>=.55).all(1)&sample["detected"]&~sample["ambiguous"]&np.isfinite(sig[key])
        series=smooth(sig[key],valid,1)
        reps=repetitions(series,times)
        amplitude=(q(series,95) or 0)-(q(series,5) or 0)
        tracks.append(dict(node=node,key=key,series=series,valid=valid,reps=reps,amplitude=amplitude))
    active=max(tracks,key=lambda t:t["amplitude"]*t["valid"].mean())
    rows=[]
    for index,rep in enumerate(active["reps"],1):
        a,b,p=rep["start"],rep["end"],rep["peak"]
        left=tracks[0]["series"][a:b];right=tracks[1]["series"][a:b]
        both=np.isfinite(left)&np.isfinite(right)
        diff=abs(left[both]-right[both]);den=(abs(left[both])+abs(right[both]))/2
        asym=float(np.median(diff/np.maximum(den,10))*100) if both.sum()>=3 else None
        values=active["series"][a:b]
        # Fases cinemáticas; no equivalen automáticamente a fases musculares.
        increasing=float(times[p]-times[a]);decreasing=float(times[b-1]-times[p])
        concentric=decreasing if exercise in ["squat","push_up"] else increasing
        eccentric=increasing if exercise in ["squat","push_up"] else decreasing
        rows.append(dict(repetition=index,start_s=float(times[a]),peak_s=float(times[p]),end_s=float(times[b-1]),
            complete=bool(rep["complete"]),rom_deg=(q(values,95) or 0)-(q(values,5) or 0),peak_deg=q(values,95),
            concentric_s=concentric,eccentric_s=eccentric,bilateral_asymmetry_pct=asym,
            asymmetry_kind="diferencia angular simultánea; curl alternado no implica defecto",primary=active["key"],
            phase_status="asignación mecánica aproximada; sin referencia de fuerza",label_scope="video, no repetición"))
    return rows,float(active["valid"].mean())

def generic_features(x,mask):
    # Resumen angular/espacial fijo para baselines de los ejercicios nuevos.
    result=[]
    for node in range(33):
        for channel in range(8):
            values=x[:,node,channel][mask[:,node,channel]]
            result.extend(np.percentile(values,[10,50,90]).tolist()+[float(values.std())] if len(values) else [0]*4)
    return np.array(result,np.float32)

def one(row,cache,extractor):
    from train_camera import cached_sample
    from camera_model import features,hard_failure
    sample=cached_sample(row,cache,extractor);x,mask=encode(sample)
    reps,reading=kinematics(sample,row["exercise"])
    if row["exercise"] in ["elbow_flexion","squat"]:
        stat,r=features(sample,row["exercise"]);color=r["color"];ready=bool(r["quality"]["ready"]);hard=hard_failure(r)
    else:stat=generic_features(x,mask);color="amarillo";ready=False;hard=False
    complete=sum(r["complete"] for r in reps)
    eligible=bool(sample["coverage"]>=.6 and reading>=.5 and complete>=1 and np.mean(sample["ambiguous"])<=.2)
    return x,mask,stat,dict(**{k:row[k] for k in ["id","subject","exercise","error","view","source","original_name"]},
        coverage=sample["coverage"],reading_fraction=reading,frames=len(sample["times"]),duration_s=sample["duration"],
        complete_repetitions=complete,repetitions=len(reps),criteria_color=color,model_ready=ready,hard_failure=hard,
        eligible=eligible,reason="incluido" if eligible else "cobertura <0.6, lectura <0.5, ambigüedad >0.2 o sin ciclo completo"),reps

def prepare(manifest,cache,out,workers=2):
    import pandas as pd
    from pose import extractor_id,FPS
    from camera_model import signature
    out=Path(out);out.mkdir(parents=True,exist_ok=True);source=json.loads(Path(manifest).read_text());rows=source["rows"]
    extractor=extractor_id();version=signature();all_records=[];all_reps=[];arrays={};failures=[]
    context=multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        futures={pool.submit(one,r,cache,extractor):r for r in rows}
        for i,f in enumerate(as_completed(futures),1):
            row=futures[f]
            try:
                x,m,s,record,reps=f.result();all_records.append(record)
                all_reps.extend(dict(id=row["id"],subject=row["subject"],exercise=row["exercise"],**r) for r in reps)
                if record["eligible"]:arrays.setdefault(row["exercise"],[]).append((record,x,m,s))
            except Exception as exc:
                record=dict(row,eligible=False,reason=f"extracción: {type(exc).__name__}: {exc}")
                record.pop("path",None);all_records.append(record);failures.append(record)
            if i%10==0 or i==len(rows):print(f"Pose RGB: {i}/{len(rows)}; fallos: {len(failures)}",flush=True)
    audit=pd.DataFrame(all_records).sort_values(["exercise","id"]);audit.to_csv(out/"auditoria_dataset.csv",index=False)
    pd.DataFrame(all_reps).to_csv(out/"cinematica_repeticiones.csv",index=False)
    hashes={}
    for exercise,items in arrays.items():
        items.sort(key=lambda a:a[0]["id"]);records=[a[0] for a in items];path=out/f"{exercise}.npz"
        np.savez_compressed(path,x=np.stack([a[1] for a in items]),mask=np.stack([a[2] for a in items]),
            stat=np.stack([a[3] for a in items]),metadata=json.dumps(records,ensure_ascii=False))
        hashes[exercise]=hashlib.sha256(path.read_bytes()).hexdigest()
    metadata=dict(dataset_doi=source["doi"],dataset_version=source["version"],license=source["license"],attribution=source["attribution"],
        extractor_id=extractor,pipeline_signature=version,representation=fingerprint(),target_pose_fps=FPS,sequence_steps=64,
        downloaded=len(rows),included=int(audit.eligible.sum()),excluded=int((~audit.eligible).sum()),files_sha256=hashes,
        duplicates=len(source["duplicates"]),conflicts=len(source["conflicts_rejected"]),limit_per_exercise=source["limit_per_exercise"],
        labels="0 good / 1 bad por video, asignadas por los autores; sin error tipificado ni anotaciones por repetición")
    (out/"data.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    audit.groupby(["exercise","error","view","eligible"],dropna=False).size().reset_index(name="videos").to_csv(out/"distribucion_dataset.csv",index=False)
    if not hashes:raise RuntimeError("Ningún ejercicio tiene videos elegibles; consultar auditoría.")
    print(json.dumps(metadata,ensure_ascii=False,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--cache",required=True);p.add_argument("--out",required=True)
    p.add_argument("--workers",type=int,default=2);a=p.parse_args();prepare(a.manifest,a.cache,a.out,a.workers)
