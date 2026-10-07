"""MediaPipe Tasks: extracción idéntica para aprendizaje y evaluación con cámara."""
import hashlib
import os
from pathlib import Path
import threading
import numpy as np

POSE_URL='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
_lock=threading.Lock();_detector=None


def detector():
    global _detector
    if _detector is None:
        import mediapipe as mp
        import requests
        p=Path(os.environ.get('MOVEAI_ASSETS_DIR',Path(__file__).parent/'assets'))/'pose_landmarker_lite.task'
        p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or p.stat().st_size<1_000_000:
            temp=p.with_suffix('.download')
            with requests.get(POSE_URL,stream=True,timeout=(20,180)) as r:
                r.raise_for_status()
                with temp.open('wb') as f:
                    for chunk in r.iter_content(1024*1024):f.write(chunk)
            if temp.stat().st_size<1_000_000:raise ValueError('El detector no terminó de descargar.')
            os.replace(temp,p)
        opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(p)),
              running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=2,
              min_pose_detection_confidence=.5,min_pose_presence_confidence=.5)
        _detector=mp.tasks.vision.PoseLandmarker.create_from_options(opts)
    return _detector


def fill(arr):
    out=arr.copy();flat=out.reshape(len(out),-1)
    for j in range(flat.shape[1]):
        valid=np.isfinite(flat[:,j])
        if valid.any():flat[:,j]=np.interp(np.arange(len(out)),np.where(valid)[0],flat[valid,j])
        else:flat[:,j]=0
    return out


def extract_video(path):
    import cv2
    import mediapipe as mp
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError('No se pudo abrir el video.')
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        if n<8:raise ValueError('El video es demasiado corto o no tiene fotogramas.')
        if fps and n/fps>60:raise ValueError('Recorta el video a una repetición de hasta 60 segundos.')
        world=np.full((64,33,3),np.nan,np.float32);im=np.full((64,33,2),np.nan,np.float32)
        vis=np.zeros((64,33),np.float32);detected=np.zeros(64,bool);multiple=0;thumb=None
        with _lock:
            det=detector()
            for pos,frame_idx in enumerate(np.linspace(0,n-1,64).round().astype(int)):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_idx));ok,frame=cap.read()
                if not ok:continue
                hh,ww=frame.shape[:2];scale=min(1,640/max(hh,ww))
                small=cv2.resize(frame,(round(ww*scale),round(hh*scale)))
                rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
                res=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                if len(res.pose_landmarks)>1:multiple+=1;continue
                if not res.pose_landmarks or not res.pose_world_landmarks:continue
                detected[pos]=True
                world[pos]=[[p.x,p.y,p.z] for p in res.pose_world_landmarks[0]]
                im[pos]=[[p.x*(ww/hh),p.y] for p in res.pose_landmarks[0]]
                vis[pos]=[getattr(p,'visibility',0) or 0 for p in res.pose_landmarks[0]]
                if thumb is None:thumb=rgb
        world=fill(world);im=fill(im)
        hip=(world[:,23]+world[:,24])/2;sh=(world[:,11]+world[:,12])/2
        scale=np.maximum(np.median(np.linalg.norm(sh-hip,axis=1)),1e-3)
        world=(world-hip[:,None,:])/scale
        return {'world':world,'image':im,'visibility':vis,'coverage':float(detected.mean()),
                'multiple_people':multiple,'thumbnail':thumb,'pose_version':mp.__version__}
    finally:cap.release()
