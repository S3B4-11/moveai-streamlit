"""Colección supervisada, deduplicada y trazable; nunca aprende de sus propias predicciones."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
import numpy as np
from geometry import EXERCISES, TAGS


def data_root():
    p=Path(os.environ.get('MOVEAI_DATA_DIR',Path(__file__).parent/'datos_privados'))
    p.mkdir(parents=True,exist_ok=True);return p


def connect(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(root/'feedback.sqlite',timeout=30)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, subject_id TEXT, exercise TEXT, is_correct INTEGER, tags TEXT, reviewer TEXT, reviewed INTEGER, metadata TEXT, created TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS revisions (id TEXT, data TEXT, created TEXT)')
    return conn


def save_review(video_path, sample, exercise, subject_id, is_correct, tags, reviewer, consent, reviewed, root=None):
    if not consent or not reviewed:
        raise ValueError('Confirma permiso y revisión humana antes de guardar.')
    if exercise not in EXERCISES or is_correct not in (0,1,False,True):
        raise ValueError('Ejercicio o etiqueta inválida.')
    subject_id=subject_id.strip();reviewer=reviewer.strip();tags=sorted(set(tags or []))
    if not subject_id or not reviewer:
        raise ValueError('Indica el mismo alias de persona en todos sus videos y un revisor.')
    if not set(tags)<=set(TAGS[exercise]) or (is_correct and tags):
        raise ValueError('Las etiquetas contradicen el ejercicio o la ejecución correcta.')
    if not sample.get('trainable',False):
        raise ValueError('Este video no tiene suficientes articulaciones visibles para entrenar.')
    world=np.asarray(sample['world'],np.float32);image=np.asarray(sample['image'],np.float32)
    visibility=np.asarray(sample['visibility'],np.float32)
    if world.shape!=(64,33,3) or image.shape!=(64,33,2) or visibility.shape!=(64,33) or not np.isfinite(world).all():
        raise ValueError('Secuencia inválida.')
    path=Path(video_path)
    if path.stat().st_size>150*1024*1024:raise ValueError('Video demasiado grande; máximo 150 MB.')
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    sid=digest.hexdigest();root=Path(root or data_root());(root/'samples').mkdir(parents=True,exist_ok=True)
    dest=root/'samples'/f'{sid}.npz'
    metadata={k:sample[k] for k in ['view','side','target','coverage','pose_version'] if k in sample}
    now=datetime.now(timezone.utc).isoformat()
    with connect(root) as conn:
        old=conn.execute('SELECT subject_id,exercise FROM samples WHERE id=?',(sid,)).fetchone()
        if old and old!=(subject_id,exercise):
            raise ValueError('Ese mismo video ya está registrado para otra persona o ejercicio.')
        if not dest.exists():
            fd,tmp=tempfile.mkstemp(suffix='.npz',dir=root/'samples');os.close(fd)
            try:
                np.savez_compressed(tmp,world=world,image=image,visibility=visibility)
                os.replace(tmp,dest)
            finally:
                if Path(tmp).exists():Path(tmp).unlink()
        row=(sid,subject_id,exercise,int(is_correct),json.dumps(tags),reviewer,1,json.dumps(metadata),now)
        conn.execute('INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?,?,?,?)',row)
        conn.execute('INSERT INTO revisions VALUES (?,?,?)',(sid,json.dumps(row),now))
    return sid


def export_reviews(root=None):
    root=Path(root or data_root())
    with connect(root) as c:
        rows=c.execute('SELECT * FROM samples WHERE reviewed=1 ORDER BY id').fetchall()
    if not rows:raise ValueError('Todavía no hay videos revisados.')
    dest=root/'moveai_videos_revisados.zip'
    buf=io.StringIO();writer=csv.writer(buf)
    writer.writerow(['id','subject_id','exercise','is_correct','tags','reviewer','reviewed','metadata','created','sequence_sha256'])
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for row in rows:
            p=root/'samples'/f'{row[0]}.npz'
            writer.writerow([*row,hashlib.sha256(p.read_bytes()).hexdigest()])
            z.write(p,f'samples/{p.name}')
        z.writestr('labels.csv',buf.getvalue())
        z.writestr('schema.json',json.dumps({'version':1,'labels':'human_review','raw_video_stored':False}))
    return str(dest)


def load_reviews(root=None):
    root=Path(root or data_root());items=[]
    with connect(root) as conn:
        rows=conn.execute('SELECT * FROM samples WHERE reviewed=1 ORDER BY id').fetchall()
    for row in rows:
        with np.load(root/'samples'/f'{row[0]}.npz',allow_pickle=False) as f:
            world=f['world'].copy()
        items.append({'id':row[0],'subject':'camera:'+row[1],'exercise':row[2], 'error':1-row[3],
                      'tags':json.loads(row[4]),'coords':world,'source':'camera','metadata':json.loads(row[7])})
    return items


def import_reviews(zip_path,root=None):
    """Importa exportaciones sin extraer rutas arbitrarias ni ejecutar objetos serializados."""
    root=Path(root or data_root());(root/'samples').mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        if sum(a.file_size for a in z.infolist())>1024**3:raise ValueError('Paquete demasiado grande.')
        rows=list(csv.DictReader(io.StringIO(z.read('labels.csv').decode('utf-8'))))
        for row in rows:
            sid=row['id']
            if len(sid)!=64 or any(c not in '0123456789abcdef' for c in sid):raise ValueError('Identificador inválido.')
            if row['exercise'] not in EXERCISES or row['is_correct'] not in ['0','1'] or row['reviewed']!='1':raise ValueError('Etiqueta inválida.')
            tags=json.loads(row['tags'])
            if not set(tags)<=set(TAGS[row['exercise']]) or (row['is_correct']=='1' and tags):raise ValueError('Subtipos inválidos.')
            raw=z.read(f'samples/{sid}.npz')
            if hashlib.sha256(raw).hexdigest()!=row['sequence_sha256']:raise ValueError('Integridad inválida.')
            with np.load(io.BytesIO(raw),allow_pickle=False) as f:
                if f['world'].shape!=(64,33,3) or not np.isfinite(f['world']).all():raise ValueError('Secuencia inválida.')
            dest=root/'samples'/f'{sid}.npz'
            with connect(root) as conn:
                old=conn.execute('SELECT subject_id,exercise FROM samples WHERE id=?',(sid,)).fetchone()
                if old and old!=(row['subject_id'],row['exercise']):raise ValueError('Video duplicado con otra identidad.')
                if not dest.exists():dest.write_bytes(raw)
                values=[row[k] for k in ['id','subject_id','exercise','is_correct','tags','reviewer','reviewed','metadata','created']]
                conn.execute('INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?,?,?,?)',values)
    return len(rows)
