"""Descarga automática, selectiva, reanudable y auditada de Mendeley V3."""
import argparse
import hashlib
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
import requests
from rgb_dataset import HTTPRangeFile,API,PAGE,DOI

EXERCISES=["elbow_flexion","squat","push_up","shoulder_press"]

def label(name):
    text=name.lower().replace("\\","/")
    exercise="elbow_flexion" if re.search(r"bicep|biceps|curl",text) else "squat" if "squat" in text else \
        "push_up" if re.search(r"push[_\-\s]?up",text) else "shoulder_press" if re.search(r"(?:^|[/_])shoulder(?:[/_]|$)",text) else None
    quality=set(re.findall(r"(?:^|[/_\-\s])(good|bad)(?=[/_\-\s.]|$)",text))
    subject=re.search(r"(?:subject|person|participant|sub|s)[_\-\s]*(\d+)",text)
    view=re.findall(r"(?:^|[/_\-\s])(front|side|diagonal)(?=[/_\-\s.]|$)",text)
    if exercise is None or len(quality)!=1 or subject is None:return None
    return dict(exercise=exercise,error=int(next(iter(quality))=="bad"),subject="mendeley_rgb:"+str(int(subject.group(1))),
        view=view[0] if view else "unknown",source="mendeley_rgb")

class RangeUnsupported(ValueError):pass

class RetryingRange(HTTPRangeFile):
    def read(self,n=-1):
        for attempt in range(4):
            position=self.position
            try:return super().read(n)
            except ValueError as exc:
                self.position=position
                if "no admite rangos" in str(exc):raise RangeUnsupported(str(exc)) from exc
                if attempt==3:raise
            except requests.RequestException:
                self.position=position
                if attempt==3:raise
            time.sleep(min(2**attempt,8))

def full_archive(url,size,destination):
    destination=Path(destination)
    if destination.exists() and destination.stat().st_size==size:return destination
    if shutil.disk_usage(destination.parent).free<size*1.2:raise RuntimeError("Espacio insuficiente para el ZIP completo. Libera espacio o usa una carpeta de Drive con cuota disponible.")
    temp=destination.with_suffix(".part")
    for attempt in range(4):
        begin=temp.stat().st_size if temp.exists() else 0
        try:
            with requests.get(url,headers={"Range":f"bytes={begin}-"} if begin else {},stream=True,timeout=(30,120)) as r:
                r.raise_for_status()
                if begin and r.status_code!=206:begin=0
                with temp.open("ab" if begin else "wb") as dest:
                    for chunk in r.iter_content(4*1024*1024):dest.write(chunk)
            if temp.stat().st_size!=size:raise ValueError("ZIP incompleto.")
            temp.replace(destination);return destination
        except (requests.RequestException,ValueError):
            if attempt==3:raise
            time.sleep(2**attempt)

def obtain(root,limit=0,exercises=EXERCISES,archive=None):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    response=requests.get(API,params={"version":3,"folder_id":"root"},timeout=60);response.raise_for_status()
    api=response.json();(root/"dataset_api_snapshot.json").write_text(json.dumps(api,indent=2))
    entries=[e for e in api if e["filename"]=="Dataset Exercise Quality-wise.zip"]
    if len(entries)!=1:raise ValueError("Cambió la organización oficial del dataset; revisar API antes de entrenar.")
    info=entries[0]["content_details"];size=int(info["size"])
    if archive:remote=Path(archive)
    else:remote=RetryingRange(info["download_url"],size)
    try:z=zipfile.ZipFile(remote)
    except RangeUnsupported:
        print("Rangos no disponibles: descarga automática del ZIP oficial completo.",flush=True)
        remote=full_archive(info["download_url"],size,root/"dataset_official.zip");z=zipfile.ZipFile(remote)
    rows=[];seen={};conflicts=set();duplicates=[];counts=dict.fromkeys(exercises,0)
    with z:
        candidates=[e for e in z.infolist() if e.filename.lower().endswith(".mp4") and label(e.filename) and label(e.filename)["exercise"] in exercises]
        # En modo rápido mezcla personas, clases y vistas de forma determinista.
        candidates.sort(key=lambda e:hashlib.sha256(e.filename.encode()).hexdigest())
        for entry in candidates:
            meta=label(entry.filename);exercise=meta["exercise"]
            if limit and counts[exercise]>=limit:continue
            if entry.file_size>250*1024*1024:raise ValueError("Video individual excede 250 MB.")
            path=root/"videos"/exercise/(hashlib.sha256(entry.filename.encode()).hexdigest()+".mp4")
            path.parent.mkdir(parents=True,exist_ok=True)
            # Verificación CRC: reanudación también detecta archivos locales alterados.
            import zlib
            valid=path.exists() and path.stat().st_size==entry.file_size
            if valid:
                crc=0
                with path.open("rb") as source:
                    for chunk in iter(lambda:source.read(1024*1024),b""):crc=zlib.crc32(chunk,crc)
                valid=(crc&0xffffffff)==entry.CRC
            if not valid:
                for attempt in range(4):
                    temp=path.with_suffix(".part")
                    try:
                        with z.open(entry) as source,temp.open("wb") as dest:
                            shutil.copyfileobj(source,dest,1024*1024)
                        if temp.stat().st_size!=entry.file_size:raise ValueError("Video incompleto.")
                        temp.replace(path);break
                    except (requests.RequestException,zipfile.BadZipFile,ValueError):
                        if attempt==3:raise
                        time.sleep(2**attempt)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in conflicts:continue
            if digest in seen:
                previous=seen[digest];duplicates.append(dict(id=digest,first=previous["original_name"],duplicate=entry.filename))
                if any(previous[k]!=meta[k] for k in ["exercise","subject","error"]):
                    conflicts.add(digest);rows=[r for r in rows if r["id"]!=digest];counts[previous["exercise"]]-=1
                continue
            row=dict(meta,id=digest,path=str(path),original_name=entry.filename,bytes=entry.file_size)
            rows.append(row);seen[digest]=row;counts[exercise]+=1
            print(f"RGB {exercise}: {counts[exercise]} videos; {meta['subject']}; {'bad' if meta['error'] else 'good'}",flush=True)
    if not rows:raise ValueError("No hay videos con etiquetas verificables.")
    result=dict(rows=rows,page=PAGE,doi=DOI,version=3,license="CC BY 4.0",
        attribution="Duddela Sai Prashanth, Priya Kamath, Rakshith Bhandary, S Ashlesh Pai, Pratham PN, Advith P y Abhishek Mashetty (2026). Mendeley Data V3.",
        duplicates=duplicates,conflicts_rejected=sorted(conflicts),archive_bytes=size,limit_per_exercise=limit,
        official_candidates=len(candidates),counts=counts)
    (root/"manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--limit",type=int,default=0)
    p.add_argument("--exercises",nargs="+",default=EXERCISES);p.add_argument("--archive")
    a=p.parse_args();obtain(a.root,a.limit,a.exercises,a.archive)
