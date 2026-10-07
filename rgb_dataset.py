"""Descarga selectiva del dataset RGB CC BY 4.0 de Mendeley."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import requests

PAGE="https://data.mendeley.com/datasets/kgbb3yn47p/3"
DOI="10.17632/kgbb3yn47p.3"
API="https://data.mendeley.com/public-api/datasets/kgbb3yn47p/files"

class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url=url;self.size=int(size);self.position=0;self.session=requests.Session()
        self.cached_start=0;self.cached=b""
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError("Offset inválido.")
        return self.position
    def read(self,n=-1):
        n=min(self.size-self.position,n if n>=0 else self.size-self.position)
        if n<=0:return b""
        start=self.position
        if self.cached_start<=start and start+n<=self.cached_start+len(self.cached):
            begin=start-self.cached_start;self.position+=n;return self.cached[begin:begin+n]
        end=min(self.size-1,start+max(n,4*1024*1024)-1)
        with self.session.get(self.url,headers={"Range":f"bytes={start}-{end}"},timeout=(20,90),stream=True) as r:
            r.raise_for_status()
            if r.status_code!=206:raise ValueError("El servidor no admite rangos; descarga el ZIP oficial y usa --archive.")
            data=r.content
        if len(data)!=end-start+1:raise ValueError("Rango incompleto.")
        self.cached_start=start;self.cached=data;self.position+=n
        return data[:n]

def parse_label(name):
    name=name.lower().replace("\\","/")
    exercise="elbow_flexion" if re.search("bicep|biceps|curl",name) else "squat" if "squat" in name else None
    if exercise is None:return None
    quality=re.findall(r"(?:^|[/_\-\s])(good|bad)(?=[/_\-\s.]|$)",name)
    view=re.findall(r"(?:^|[/_\-\s])(front|side|diagonal)(?=[/_\-\s.]|$)",name)
    person=re.search(r"(?:subject|person|participant|sub|s)[_\-\s]*(\d+)",name)
    if person is None or len(set(quality))!=1:return None
    return dict(exercise=exercise,error=int(quality[0]=="bad"),subject="mendeley_rgb:"+str(int(person.group(1))),
                view=view[0] if view else "unknown",source="mendeley_rgb")

def obtain(root,archive=None,limit=0):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if archive:remote=Path(archive)
    else:
        r=requests.get(API,params={"version":3,"folder_id":"root"},timeout=45);r.raise_for_status()
        entries=[e for e in r.json() if e["filename"]=="Dataset Exercise Quality-wise.zip"]
        if len(entries)!=1:raise ValueError("No se encontró la organización esperada del dataset.")
        details=entries[0]["content_details"];remote=HTTPRangeFile(details["download_url"],details["size"])
    rows=[];seen={};conflicts=set();duplicates=[];counts={"elbow_flexion":0,"squat":0}
    with zipfile.ZipFile(remote) as z:
        candidates=[e for e in z.infolist() if e.filename.lower().endswith(".mp4") and parse_label(e.filename)]
        def order(entry):
            label=parse_label(entry.filename)
            return ({"front":0,"side":1,"diagonal":2}.get(label["view"],3),int(label["subject"].split(":")[-1]),label["error"])
        for entry in sorted(candidates,key=order):
            label=parse_label(entry.filename);exercise=label["exercise"]
            if limit and counts[exercise]>=limit:continue
            if entry.file_size>250*1024*1024:raise ValueError("Video individual mayor de 250 MB.")
            path=root/"videos"/exercise/(hashlib.sha256(entry.filename.encode()).hexdigest()+".mp4");path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists() or path.stat().st_size!=entry.file_size:
                temp=path.with_suffix(".part")
                with z.open(entry) as source,temp.open("wb") as dest:
                    for chunk in iter(lambda:source.read(1024*1024),b""):dest.write(chunk)
                if temp.stat().st_size!=entry.file_size:raise ValueError("Video incompleto.")
                temp.replace(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in conflicts:continue
            if digest in seen:
                previous=seen[digest];duplicates.append(dict(id=digest,first=previous["original_name"],duplicate=entry.filename))
                if any(previous[k]!=label[k] for k in ["exercise","subject","error"]):
                    conflicts.add(digest);rows=[row for row in rows if row["id"]!=digest];counts[previous["exercise"]]-=1
                continue
            row=dict(label,id=digest,path=str(path.resolve()),original_name=entry.filename)
            seen[digest]=row;rows.append(row);counts[exercise]+=1
            print("Descargado",exercise,counts[exercise],label["subject"],flush=True)
    manifest=dict(page=PAGE,doi=DOI,license="CC BY 4.0",attribution="Duddela Sai Prashanth et al. (2026), Mendeley Data V3",
                  rows=rows,duplicates=duplicates,conflicts_rejected=sorted(conflicts))
    (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if not rows:raise ValueError("No se encontraron videos y etiquetas interpretable.")
    return manifest

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",default="rgb_data")
    parser.add_argument("--archive");parser.add_argument("--limit",type=int,default=0)
    args=parser.parse_args();obtain(args.root,args.archive,args.limit)
