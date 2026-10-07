"""Datos de aprendizaje revisados por personas, sin autoetiquetado."""
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from geometry import SCHEMA,EXERCISES

def save_review(video,sample,exercise,subject,correct,reviewer,consent,reviewed,root):
    if not consent or not reviewed:raise ValueError("Confirma la revisión del video y el permiso de la persona.")
    if not subject.strip() or not reviewer.strip():raise ValueError("Indica el alias y quién revisó.")
    if not sample.get("trainable"):raise ValueError("La lectura no es suficiente para añadir entrenamiento.")
    if exercise not in EXERCISES:raise ValueError("Ejercicio desconocido.")
    root=Path(root);(root/"samples").mkdir(parents=True,exist_ok=True)
    sid=hashlib.sha256(Path(video).read_bytes()).hexdigest()
    labels_path=root/"labels.json";labels=json.loads(labels_path.read_text()) if labels_path.exists() else {}
    if sid in labels and (labels[sid]["subject"]!=subject.strip() or labels[sid]["exercise"]!=exercise):
        raise ValueError("El mismo video ya está asociado a otra persona o ejercicio.")
    np.savez_compressed(root/"samples"/(sid+".npz"),**{k:sample[k] for k in ["world","image","visibility","times","detected","ambiguous"]})
    labels[sid]=dict(id=sid,subject=subject.strip(),exercise=exercise,error=int(not correct),reviewer=reviewer.strip(),
                    reviewed_at=datetime.now(timezone.utc).isoformat(),
                    metadata={k:sample[k] for k in ["pose_version","extractor_id","pipeline","coverage","target","side"]})
    labels_path.write_text(json.dumps(labels,ensure_ascii=False,indent=2));return sid

def export_reviews(root):
    root=Path(root);labels=json.loads((root/"labels.json").read_text());buf=io.StringIO();writer=csv.writer(buf)
    writer.writerow(["id","subject","exercise","error","reviewer","reviewed_at","metadata","sha256"])
    dest=root/"moveai_revisiones_rgb.zip"
    with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("schema.json",json.dumps(dict(schema=SCHEMA,origin="human_review",raw_video=False)))
        for row in labels.values():
            p=root/"samples"/(row["id"]+".npz");z.write(p,"samples/"+p.name)
            writer.writerow([row[k] for k in ["id","subject","exercise","error","reviewer","reviewed_at"]]+
                            [json.dumps(row["metadata"]),hashlib.sha256(p.read_bytes()).hexdigest()])
        z.writestr("labels.csv",buf.getvalue())
    return dest

def review_rows(path,expected_extractor):
    rows=[]
    with zipfile.ZipFile(path) as z:
        if sum(e.file_size for e in z.infolist())>512*1024*1024:raise ValueError("Revisiones mayores de 512 MB.")
        schema=json.loads(z.read("schema.json"))
        if schema.get("schema")!=SCHEMA or schema.get("origin")!="human_review":raise ValueError("Formato anterior: vuelve a revisar el video con esta versión.")
        for row in csv.DictReader(io.StringIO(z.read("labels.csv").decode())):
            sid=row["id"]
            if len(sid)!=64 or any(c not in "0123456789abcdef" for c in sid):raise ValueError("ID inválido.")
            if row["exercise"] not in EXERCISES or row["error"] not in ("0","1"):raise ValueError("Etiqueta inválida.")
            meta=json.loads(row["metadata"])
            if meta.get("extractor_id")!=expected_extractor:raise ValueError("La revisión usa otra versión de pose.")
            data=z.read("samples/"+sid+".npz")
            if hashlib.sha256(data).hexdigest()!=row["sha256"]:raise ValueError("Coordenadas alteradas.")
            with np.load(io.BytesIO(data),allow_pickle=False) as f:sample={k:f[k].copy() for k in f.files}
            n=len(sample["times"])
            if sample["world"].shape!=(n,33,3) or sample["image"].shape!=(n,33,2) or sample["visibility"].shape!=(n,33) or n<8:
                raise ValueError("Secuencia de pose inválida.")
            sample.update(meta,previews=[])
            rows.append(dict(id=sid,subject="own_camera:"+row["subject"],exercise=row["exercise"],error=int(row["error"]),
                             source="own_camera",sample=sample,view="own_camera",reviewed_at=row["reviewed_at"]))
    return rows
