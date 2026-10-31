"""Comprobación funcional de reglas de press sobre MP4 y caché de poses.

No es una validación independiente ni selecciona umbrales por etiqueta.
Ejemplo: python verify_press_dataset.py --videos /ruta/shoulder --workers 2
"""
from pathlib import Path
import argparse,sys,json,hashlib,os,time,csv,concurrent.futures
import numpy as np
APP=Path(__file__).resolve().parent/'app'
if not APP.exists():APP=Path(__file__).resolve().parent
sys.path.insert(0,str(APP))

def one(args):
    video,cache,target=args
    from pose import extract_video,extractor_id
    from press_analysis import analyze_press
    from decision_policy import apply_policy
    from press_quality import finalize_press
    from moveai_core import diagnostic
    start=time.perf_counter();path=Path(video);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    file=cache/(path.stem+'.npz')
    video_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if file.exists():
        with np.load(file,allow_pickle=False) as f:
            s={k:f[k].copy() for k in ('world','image','visibility','times','detected','ambiguous')}
            meta=json.loads(str(f['metadata']))
            if meta.get('video_sha256')!=video_sha or meta.get('extractor_id')!=extractor_id():
                raise ValueError('Caché de otro video o extractor; utiliza una carpeta de caché nueva.')
            s.update(meta,previews=[])
    else:
        s=extract_video(path,previews=False)
        fields=('world','image','visibility','times','detected','ambiguous')
        meta={k:v for k,v in s.items() if k not in fields and k!='previews'}
        meta['video_sha256']=video_sha
        tmp=file.with_suffix('.tmp.npz');np.savez_compressed(tmp,**{k:s[k] for k in fields},metadata=json.dumps(meta));tmp.replace(file)
    r=analyze_press(s,target=target);r['model']={'available':False};r=finalize_press(apply_policy(r))
    return dict(file=path.name,label='incorrect' if '_bad_' in path.stem else 'correct' if '_good_' in path.stem else None,
                status=r['status'],counts=r['counts'],error_keys=sorted(set(e['key'] for e in r['errors'])),
                seconds=round(time.perf_counter()-start,3),diagnostic=diagnostic(r))

def main():
    p=argparse.ArgumentParser();p.add_argument('--videos',required=True);p.add_argument('--cache',default='press_pose_cache');p.add_argument('--output',default='press_functional_results.json');p.add_argument('--workers',type=int,default=2);p.add_argument('--target',type=float,default=150);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    files=sorted(Path(a.videos).rglob('*.mp4'))
    if a.limit:files=files[:a.limit]
    rows=[];jobs=[(str(v),a.cache,a.target) for v in files]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(one,j):j for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:rows.append(future.result())
            except Exception as e:rows.append(dict(file=Path(futures[future][0]).name,status='processing_error',reason=f'{type(e).__name__}: {e}'))
            if len(rows)%10==0:print('Procesados',len(rows),'/',len(files),flush=True)
    rows.sort(key=lambda r:r['file']);from collections import Counter
    summary=dict(n=len(rows),states=dict(Counter(r['status'] for r in rows)),
                 label_status=dict(Counter(str(r.get('label'))+' -> '+r['status'] for r in rows)),
                 scope='Prueba funcional en datos utilizados durante el desarrollo; no es una validación independiente ni certifica umbrales clínicos.',target=a.target)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(summary=summary,rows=rows),ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
