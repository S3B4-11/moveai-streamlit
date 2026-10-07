"""Importa la extracción RGB ya realizada sin videos ni rutas del ZIP."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from pose import extractor_id
from geometry import EXERCISES,SCHEMA

def import_cache(archive,destination):
    root=Path(destination).resolve();expected=extractor_id();count=0
    with zipfile.ZipFile(archive) as z:
        if sum(p.file_size for p in z.infolist())>1024*1024*1024:raise ValueError("Caché mayor de 1 GB.")
        manifest=json.loads(z.read("rgb_data/manifest.json"));ids=set()
        for row in manifest["rows"]:
            sid=row["id"]
            if len(sid)!=64 or any(c not in "0123456789abcdef" for c in sid) or sid in ids:raise ValueError("ID de video inválido o duplicado.")
            ids.add(sid)
            if row["exercise"] not in EXERCISES or row["error"] not in (0,1) or not row["subject"]:raise ValueError("Etiqueta inválida.")
            name=sid+"_"+expected[:16]+".npz"
            try:data=z.read("rgb_cache/poses/"+name)
            except KeyError:raise ValueError("Las poses no corresponden al extractor Full de esta versión. Usa el ZIP de poses actualizado.") from None
            with np.load(io.BytesIO(data),allow_pickle=False) as f:
                meta=json.loads(str(f["metadata"].item()));n=len(f["times"])
                if meta.get("extractor_id")!=expected or meta.get("pipeline")!=SCHEMA:raise ValueError("Otra versión de pose.")
                if n<8 or f["world"].shape!=(n,33,3) or f["image"].shape!=(n,33,2) or f["visibility"].shape!=(n,33):raise ValueError("Coordenadas inválidas.")
                if not np.isfinite(f["times"]).all() or not (np.diff(f["times"])>0).all():raise ValueError("Tiempos inválidos.")
            dest=root/"rgb_cache/poses"/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);count+=1
            row["path"]=str(root/"rgb_data/videos"/row["exercise"]/(sid+".mp4"))
        dest=root/"rgb_data/manifest.json";dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print("Poses listas para reutilizar:",count);return count

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("archive");p.add_argument("--destination",required=True)
    a=p.parse_args();import_cache(a.archive,a.destination)
