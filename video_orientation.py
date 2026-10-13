"""Repara una rotación del video sin modificar el extractor entrenado."""
from pathlib import Path
import shutil
import subprocess
import uuid
import numpy as np

def inferred_rotation(sample):
    p=np.asarray(sample['image'],float);vis=np.asarray(sample['visibility'],float)
    valid=np.asarray(sample['detected'],bool)&~np.asarray(sample.get('ambiguous',np.zeros(len(p))),bool)
    joints=[11,12,23,24,27,28]
    valid &= (vis[:,joints].min(axis=1)>=.6)&np.isfinite(p[:,joints]).all(axis=(1,2))
    if valid.sum()<8:return 0
    shoulders=(p[:,11]+p[:,12])/2;hips=(p[:,23]+p[:,24])/2;feet=(p[:,27]+p[:,28])/2
    def degrees(v):return np.degrees(np.arctan2(v[:,0],v[:,1]))
    body=degrees(feet-shoulders)[valid];legs=degrees(feet-hips)[valid];torso=degrees(hips-shoulders)[valid]
    radians=np.deg2rad(body)
    median=float(np.degrees(np.arctan2(np.median(np.sin(radians)),np.median(np.cos(radians)))))
    rotation=int(round(median/90))*90
    if rotation not in (-90,90,180,-180):return 0
    def distance(a):return abs((a-rotation+180)%360-180)
    if (np.mean(distance(body)<20)<.85 or np.mean(distance(legs)<35)<.8 or np.mean(distance(torso)<35)<.8):return 0
    return 180 if abs(rotation)==180 else rotation

def rotate_video(source,rotation):
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('Falta FFmpeg para corregir la orientación del video.')
    source=Path(source);destination=source.parent/('moveai_upright_'+uuid.uuid4().hex+'.mp4')
    transform={90:'transpose=1',-90:'transpose=2',180:'hflip,vflip'}[rotation]
    filters=transform+',pad=ceil(iw/2)*2:ceil(ih/2)*2'
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(source),'-vf',filters,
             '-c:v','libx264','-preset','veryfast','-crf','18','-threads','2','-pix_fmt','yuv420p',
             '-c:a','aac','-movflags','+faststart','-metadata:s:v:0','rotate=0',str(destination)]
    try:
        subprocess.run(command,check=True,capture_output=True,timeout=120)
    except Exception:
        destination.unlink(missing_ok=True);raise
    return destination
