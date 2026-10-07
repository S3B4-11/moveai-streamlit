"""Prueba reproducible de un MP4, con JSON y fotogramas anotados."""
import argparse
import json
from pathlib import Path
from moveai_core import evaluate,diagnostic
from preview import annotated

def run(video,exercise,output,side="automatica",target=None):
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    result=evaluate(video,exercise,side,target)
    (dest/"diagnostico.json").write_text(json.dumps(diagnostic(result),ensure_ascii=False,indent=2,allow_nan=False))
    for rep in result["repetitions"]:
        image=annotated(result["sample"],rep)
        if image is not None:image.save(dest/f"repeticion_{rep['number']:02d}.jpg")
    print(result["color"],"·",result["title"]);print("Repeticiones:",result["counts"])
    print("Guardado:",dest.resolve());return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("video");p.add_argument("--exercise",default="elbow_flexion")
    p.add_argument("--output",default="prueba_video");p.add_argument("--side",default="automatica");p.add_argument("--target",type=float)
    a=p.parse_args();run(a.video,a.exercise,a.output,a.side,a.target)
