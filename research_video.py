"""Análisis de un video nuevo con la red final, sin modificar la página."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from pose import extract_video
from neural_features import encode
from neural_models import load_bundle,logits,probability
from research_data import kinematics,NAMES

def evaluate(path,exercise,model_root,out):
    import pandas as pd
    started=time.perf_counter();sample=extract_video(path);x,m=encode(sample)
    model,norm,meta=load_bundle(Path(model_root)/exercise)
    if meta["extractor_id"]!=sample["extractor_id"]:raise ValueError("Extractor incompatible con la red final.")
    p=float(probability(logits(model,norm.transform(x[None],m[None])),meta["calibration"])[0])
    reps,reading=kinematics(sample,exercise);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=dict(exercise=exercise,name=NAMES[exercise],family=meta["family"],p_error=p,p_correct=1-p,
        reading_fraction=reading,coverage=sample["coverage"],seconds=time.perf_counter()-started,
        approved_for_page=meta["approved"],classification="bad" if p>=.5 else "good",
        scope="probabilidad global experimental; no se asigna etiqueta a cada repetición",repetitions=reps)
    (out/"diagnostico_neuronal.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    pd.DataFrame(reps).to_csv(out/"repeticiones_neuronal.csv",index=False)
    print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--video",required=True);p.add_argument("--exercise",choices=list(NAMES),required=True)
    p.add_argument("--models",required=True);p.add_argument("--out",required=True);a=p.parse_args();evaluate(a.video,a.exercise,a.models,a.out)
