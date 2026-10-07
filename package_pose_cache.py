"""Copia portable de poses públicas extraídas, sin videos."""
import argparse
import json
from pathlib import Path
import zipfile
from pose import extractor_id

def package_cache(manifest,cache,destination):
    data=json.loads(Path(manifest).read_text());expected=extractor_id();cache=Path(cache)
    rows=[];paths=[]
    for row in data["rows"]:
        path=cache/"poses"/(row["id"]+"_"+expected[:16]+".npz")
        if not path.exists():continue
        rows.append(dict(row,path="rgb_data/videos/"+row["exercise"]+"/"+Path(row["path"]).name));paths.append(path)
    data.update(rows=rows,pose_extractor_id=expected,poses_packaged=len(rows),missing_poses_rejected=len(data["rows"])-len(rows))
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path,"rgb_cache/poses/"+path.name)
        z.writestr("rgb_data/manifest.json",json.dumps(data,ensure_ascii=False,indent=2))
        z.writestr("LEEME.txt","Coordenadas derivadas de Mendeley Data V3, Duddela Sai Prashanth et al. (2026), DOI 10.17632/kgbb3yn47p.3, CC BY 4.0. No contiene videos. Reutilizar con MOVEAI RGB 4 / MediaPipe Full 0.10.21 y conservar los nombres de archivo. Importar desde el Colab con IMPORTAR_POSES_GUARDADAS=True.\n")
    print("Poses empaquetadas:",len(rows),"·",destination.resolve());return destination

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--cache",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();package_cache(a.manifest,a.cache,a.output)
