"""Fuentes originales con etiqueta binaria real. No se inventan subtipos."""
import hashlib
import io
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

RECORD='13305826'
FILES={'3d_joints.zip':'c75ae3fc13fcf16d7f4ca36b93849c74',
       'Segmentation.csv':'90b8fbd7445dd050bf27b17126c78fbe',
       'joints_names.txt':'3f04a37cce43086b1ea1fa49f1e4514f',
       'Segmentation.txt':'5f2a5b886c6f794f03e1a8f642738c86'}
MOVEMENTS={1:'shoulder_abduction',5:'inline_lunge',6:'squat'}
MAP={11:7,12:12,13:8,14:13,15:9,16:14,23:16,24:21,25:17,26:22,27:18,28:23}


def checksum(p):
    h=hashlib.md5()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def download_rehab(root):
    """551 MB, verificación MD5 oficial, descarga atómica y reuso en Drive."""
    import requests
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for name,md5 in FILES.items():
        p=root/name
        if p.exists() and checksum(p)==md5:continue
        url=f'https://zenodo.org/records/{RECORD}/files/{name}?download=1'
        tmp=p.with_suffix(p.suffix+'.partial');print('Descargando REHAB24-6:',name,flush=True)
        # Se reintenta una descarga fallida, nunca se usa un ZIP parcial.
        for attempt in range(3):
            try:
                with requests.get(url,stream=True,timeout=(30,180)) as r:
                    r.raise_for_status()
                    with tmp.open('wb') as f:
                        for chunk in r.iter_content(4*1024*1024):f.write(chunk)
                if checksum(tmp)!=md5:raise ValueError('Checksum incorrecto en '+name)
                tmp.replace(p);break
            except (requests.RequestException,ValueError):
                if attempt==2:raise
    (root/'provenance.json').write_text(__import__('json').dumps({
        'url':f'https://zenodo.org/records/{RECORD}', 'license_text':
        'Academic or non-profit organization noncommercial research use only; cite Cernek et al. SISAP 2024.',
        'checksums':FILES,'mapping':MAP,'correctness':'1 original = correcto; 0 = incorrecto; subtipos desconocidos'},indent=2))
    return root


def resample(c,n=64):
    idx=np.linspace(0,len(c)-1,n)
    out=np.stack([np.interp(idx,np.arange(len(c)),c[:,j,k]) for j in range(33) for k in range(3)],axis=1)
    return out.reshape(n,33,3).astype(np.float32)


def load_rehab(root):
    root=Path(root);ann=pd.read_csv(root/'Segmentation.csv',sep=';');items=[];rejected=[]
    with zipfile.ZipFile(root/'3d_joints.zip') as z:
        names=set(z.namelist())
        for (ex,video),rows in ann[ann.exercise_id.isin(MOVEMENTS)].groupby(['exercise_id','video_id']):
            name=f'Ex{ex}/{video}-30fps.npy'
            if name not in names:raise ValueError('Falta '+name)
            raw=np.load(io.BytesIO(z.read(name)),allow_pickle=False)
            if raw.ndim!=3 or raw.shape[1]!=26 or raw.shape[2]<3:raise ValueError('Formato inesperado: '+name)
            for row in rows.itertuples():
                rid=f'rehab246:{video}:{row.repetition_number}'
                if row.mocap_erroneous!=0:rejected.append(rid);continue
                start,end=int(row.first_frame),int(row.last_frame)
                if not 0<=start<end<len(raw):
                    rejected.append(rid+':segmentation_bounds');continue
                coords=np.zeros((end-start+1,33,3),np.float32)
                for dst,src in MAP.items():coords[:,dst]=raw[start:end+1,src,:3]
                if not np.isfinite(coords).all():rejected.append(rid);continue
                coords[:,:,1]*=-1 # +Y hacia abajo, igual que MediaPipe.
                hip=(coords[:,23]+coords[:,24])/2;sh=(coords[:,11]+coords[:,12])/2
                coords=(coords-hip[:,None])/max(float(np.median(np.linalg.norm(sh-hip,axis=1))),1e-3)
                items.append({'id':rid,'subject':f'rehab246:P{row.person_id:02d}',
                    'exercise':MOVEMENTS[ex],'error':1-int(row.correctness),'tags':None,
                    'coords':resample(coords),'source':'rehab246','metadata':{'orientation':row.cam17_orientation}})
    print(f'REHAB24-6: {len(items)} repeticiones compatibles; {len(rejected)} con mocap erróneo o segmentación fuera de límites descartadas.')
    return items


def load_legacy_cache(root):
    """Reutiliza X/y del Colab anterior. Subtipos derivados se ignoran; y>0 conserva la etiqueta binaria."""
    root=Path(root);items=[]
    for ex in ['squat','inline_lunge','shoulder_abduction','elbow_flexion']:
        paths=sorted(root.glob(f'{ex}_v*.npz'),key=lambda p:p.stat().st_mtime,reverse=True)
        if not paths:print('Sin caché anterior:',ex);continue
        with np.load(paths[0],allow_pickle=False) as f:
            x,y,subjects=f['X'],f['y'],f['subject_ids']
        seen={}
        for i in range(len(y)):
            subj=str(subjects[i]);source=subj.split('_')[0]
            if source not in ['irds','uiprmdA01','uiprmdA03','uiprmdA07','uiprmdm01','uiprmdm03','uiprmdm07']:
                continue # No se reimportan datos sintéticos ni propias sin revisión trazable.
            coords=x[i,:,:99].reshape(64,33,3)
            rid=hashlib.sha256(coords.tobytes()).hexdigest()
            if rid in seen:
                if seen[rid] != int(y[i]>0):raise ValueError('Caché con etiquetas contradictorias.')
                continue
            seen[rid]=int(y[i]>0)
            parts=subj.split('_');person=parts[-1]
            family='irds' if source=='irds' else 'uiprmd'
            if family=='irds' and person=='217':continue
            items.append({'id':'legacy:'+rid,'subject':family+':'+person,'exercise':ex,
                          'error':int(y[i]>0),'tags':None,'coords':coords.astype(np.float32),
                          'source':family,'metadata':{'legacy_cache':paths[0].name}})
        print('Caché reutilizada:',ex,paths[0].name)
    return items

IRDS_RECORD='4610859'
IRDS_CHECKSUM='c243bcdbd1492e4928032e158accdd29'
IRDS_MAP={11:4,12:8,13:5,14:9,15:6,16:10,23:12,24:16,25:13,26:17,27:14,28:18}


def download_irds(root):
    import requests
    root=Path(root);root.mkdir(parents=True,exist_ok=True);p=root/'SkeletonData.zip'
    if p.exists() and checksum(p)==IRDS_CHECKSUM:return root
    url=f'https://zenodo.org/records/{IRDS_RECORD}/files/SkeletonData.zip?download=1'
    tmp=p.with_suffix('.partial');print('Descargando IRDS original (199 MB)...',flush=True)
    for attempt in range(3):
        try:
            with requests.get(url,stream=True,timeout=(30,180)) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(4*1024*1024):f.write(chunk)
            if checksum(tmp)!=IRDS_CHECKSUM:raise ValueError('Checksum IRDS incorrecto.')
            tmp.replace(p);break
        except (requests.RequestException,ValueError):
            if attempt==2:raise
    return root


def load_irds(root,exercises=None):
    items=[];rejected=0;exercises=set(exercises or ['elbow_flexion','shoulder_abduction'])
    gestures={0:'elbow_flexion',1:'elbow_flexion',4:'shoulder_abduction',5:'shoulder_abduction'}
    with zipfile.ZipFile(Path(root)/'SkeletonData.zip') as z:
        for name in sorted(z.namelist()):
            if '/Simplified/' not in '/'+name or not name.endswith('.txt'):continue
            bits=Path(name).stem.split('_')
            if len(bits)<5:continue
            try:g=int(bits[2])
            except ValueError:continue
            ex=gestures.get(g)
            if ex not in exercises or bits[4] not in ['1','2']:continue
            if bits[0]=='217':rejected+=1;continue # El autor indica que interrumpió los ejercicios.
            text=z.read(name).decode('utf-8-sig').strip()
            # Simplified usa CSV (algunas distribuciones usan espacios).
            first=text.splitlines()[0];delimiter=',' if ',' in first else None
            raw=np.genfromtxt(io.StringIO(text),delimiter=delimiter,dtype=np.float32)
            if raw.ndim==1:raw=raw[None,:]
            if raw.shape[1]==76 and np.isnan(raw[:,-1]).all():raw=raw[:,:75]
            if raw.shape[1]!=75:raise ValueError('IRDS: se esperaban 75 coordenadas en '+name)
            raw=raw[np.isfinite(raw).all(axis=1)]
            if len(raw)<8:rejected+=1;continue
            raw=raw.reshape(len(raw),25,3);c=np.zeros((len(raw),33,3),np.float32)
            for dst,src in IRDS_MAP.items():c[:,dst]=raw[:,src]
            c[:,:,1]*=-1;hip=(c[:,23]+c[:,24])/2;sh=(c[:,11]+c[:,12])/2
            c=(c-hip[:,None])/max(float(np.median(np.linalg.norm(sh-hip,axis=1))),1e-3)
            items.append({'id':'irds:'+Path(name).stem,'subject':'irds:'+bits[0], 'exercise':ex,
               'error':int(bits[4]=='2'),'tags':None,'coords':resample(c),'source':'irds',
               'metadata':{'position':bits[5] if len(bits)>5 else None,'session':bits[1]}})
    print(f'IRDS original: {len(items)} repeticiones; {rejected} incompletas o demasiado cortas descartadas.')
    return items


def complete_irds(items,root,legacy_archive=None):
    """Si la caché ya trae IRDS para un ejercicio, evita duplicarlo con su ZIP original."""
    need=[ex for ex in ['elbow_flexion','shoulder_abduction'] if not any(a['source']=='irds' and a['exercise']==ex for a in items)]
    if not need:return items
    root=Path(root)
    if legacy_archive and Path(legacy_archive).exists() and checksum(legacy_archive)==IRDS_CHECKSUM:
        root.mkdir(parents=True,exist_ok=True)
        import shutil
        if not (root/'SkeletonData.zip').exists():shutil.copyfile(legacy_archive,root/'SkeletonData.zip')
    return items+load_irds(download_irds(root),need)
