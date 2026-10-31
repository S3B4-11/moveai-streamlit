"""Exportaciones para tesis y aplicación, excluyendo videos y contraseñas."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

def deploy(root,run):
    root=Path(root);run=Path(run);summary=json.loads((run/"aprobacion_piloto.json").read_text())
    for item in summary:
        source=run/item["exercise"]/"final";dest=root/"neural_modelos"/item["exercise"]
        dest.mkdir(parents=True,exist_ok=True)
        for name in ["model.json","weights.npz"]:shutil.copy2(source/name,dest/name)
    ready=any(v["approved"] for v in summary)
    requirements=root/"requirements.txt";text=requirements.read_text()
    if ready and "torch==" not in text:
        text+="\n--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.6.0+cpu\nscikit-learn==1.8.0\n"
        requirements.write_text(text)
    return summary

def export(root,run,dataset,out):
    from package_app import package
    root=Path(root).resolve();run=Path(run).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    app=package(root,out/"MOVEAI_Streamlit_Neural.zip")
    result=out/"MOVEAI_Resultados_Tesis.zip"
    files=[p for p in run.rglob("*") if p.is_file() and p.suffix in [".csv",".json",".md",".txt",".png",".pdf",".html",".npz"]]
    hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/"integridad_exportacion.json").write_text(json.dumps(hashes,indent=2))
    with zipfile.ZipFile(result,"w",zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,"experimento/"+p.relative_to(run).as_posix())
        z.write(out/"integridad_exportacion.json","integridad_exportacion.json")
        for name in ["REFERENCIAS.md","METODOLOGIA.md","preservation.json"]:z.write(root/name,name)
        manifest=json.loads((Path(dataset)/"manifest.json").read_text())
        for row in manifest["rows"]:row.pop("path",None)
        z.writestr("dataset_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
        for p in root.glob("*.py"):z.write(p,"codigo/"+p.name)
    print("Aplicación:",app,"\nResultados:",result)
    return app,result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",default=Path(__file__).parent);p.add_argument("--run",required=True)
    p.add_argument("--dataset");p.add_argument("--out");p.add_argument("--deploy",action="store_true")
    a=p.parse_args()
    if a.deploy:print(json.dumps(deploy(a.root,a.run),ensure_ascii=False,indent=2))
    else:export(a.root,a.run,a.dataset,a.out)
