"""Pose RGB local de CPU, con seguimiento nuevo para cada video."""
import hashlib
import os
from pathlib import Path
import numpy as np
from geometry import SCHEMA

FPS=8
POSE_SHA="4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad"
POSE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

def asset():
    path=Path(os.environ.get("MOVEAI_ASSETS_DIR",Path(__file__).parent/"assets"))/"pose_landmarker_full.task"
    if not path.exists():
        import requests
        response=requests.get(POSE_URL,timeout=120);response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest()!=POSE_SHA:raise ValueError("Cambió el modelo de pose oficial; no mezclarlo con los clasificadores.")
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=POSE_SHA:raise ValueError("El archivo de pose no corresponde a esta versión.")
    return path

def extractor_id():
    import mediapipe as mp
    return hashlib.sha256(Path(__file__).read_bytes()+mp.__version__.encode()+asset().read_bytes()).hexdigest()

def choose(poses):
    candidates=[]
    for n,p in enumerate(poses):
        points=np.array([[v.x,v.y] for v in p]);mask=np.array([(v.visibility or 0)>=.5 for v in p])
        if mask.sum()>=6:candidates.append((float(np.prod(np.ptp(points[mask],axis=0))),n))
    candidates.sort(reverse=True)
    if not candidates:return None,False
    return candidates[0][1],len(candidates)>1 and candidates[1][0]>.6*candidates[0][0]

def extract_video(path,previews=True):
    import cv2
    import mediapipe as mp
    cv2.setNumThreads(1);cap=cv2.VideoCapture(str(path))
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps<=0 or frames<8:raise ValueError("No se puede leer una secuencia suficiente de este video.")
        if frames/fps>60.5:raise ValueError("Recorta el video a un máximo de 60 segundos.")
        stride=max(1,int(np.ceil(fps/FPS)));base=mp.tasks.BaseOptions(model_asset_path=str(asset()),delegate=mp.tasks.BaseOptions.Delegate.CPU)
        video_options=mp.tasks.vision.PoseLandmarkerOptions(base_options=base,running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1)
        audit_options=mp.tasks.vision.PoseLandmarkerOptions(base_options=base,running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=2)
        world=[];image=[];visibility=[];times=[];detected=[];ambiguous=[];thumbs=[]
        with mp.tasks.vision.PoseLandmarker.create_from_options(video_options) as detector, mp.tasks.vision.PoseLandmarker.create_from_options(audit_options) as auditor:
            index=0;last_ms=-1;last_audit=-10;multi=False;anchor=None
            while cap.grab():
                if index%stride==0:
                    ok,frame=cap.retrieve()
                    if not ok:break
                    h,w=frame.shape[:2];scale=min(1.,720/max(h,w))
                    rgb=cv2.cvtColor(cv2.resize(frame,(round(w*scale),round(h*scale))),cv2.COLOR_BGR2RGB)
                    value=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    if index/fps-last_audit>=4:
                        audit=auditor.detect(value);person,multi=choose(audit.pose_landmarks)
                        if person is not None:
                            p=audit.pose_landmarks[person];anchor=np.mean([[p[i].x,p[i].y] for i in (11,12,23,24)],axis=0)
                        last_audit=index/fps
                    ms=max(last_ms+1,round(index*1000/fps));last_ms=ms
                    result=detector.detect_for_video(value,ms);person,_=choose(result.pose_landmarks)
                    if person is not None and anchor is not None:
                        p=result.pose_landmarks[person];center=np.mean([[p[i].x,p[i].y] for i in (11,12,23,24)],axis=0)
                        if np.linalg.norm(center-anchor)>.25:person=None
                        else:anchor=.8*anchor+.2*center
                    present=person is not None and not multi
                    cw=np.full((33,3),np.nan,np.float32);ci=np.full((33,2),np.nan,np.float32);v=np.zeros(33,np.float32)
                    if present:
                        cw=np.array([[p.x,p.y,p.z] for p in result.pose_world_landmarks[person]],np.float32)
                        ci=np.array([[p.x*w/h,p.y] for p in result.pose_landmarks[person]],np.float32)
                        v=np.array([min(p.visibility or 0,p.presence if p.presence is not None else 1.) for p in result.pose_landmarks[person]],np.float32)
                    world.append(cw);image.append(ci);visibility.append(v);times.append(index/fps);detected.append(present);ambiguous.append(multi)
                    if previews and len(world)%3==1:
                        jpg=cv2.imencode(".jpg",cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,75])[1]
                        thumbs.append((len(world)-1,jpg.tobytes()))
                index+=1
        if len(world)<8:raise ValueError("No se leyó una secuencia suficiente.")
        world=np.asarray(world);good=np.asarray(detected)
        hip=(world[:,23]+world[:,24])/2;torso=np.linalg.norm((world[:,11]+world[:,12])/2-hip,axis=1)
        scale=float(np.nanmedian(torso[good])) if good.any() else 1
        if not np.isfinite(scale) or scale<.01:scale=1
        return dict(world=(world-hip[:,None,:])/scale,image=np.asarray(image),visibility=np.asarray(visibility),
                    times=np.asarray(times),detected=good,ambiguous=np.asarray(ambiguous),coverage=float(good.mean()),
                    duration=frames/fps,previews=thumbs,pose_version=mp.__version__,extractor_id=extractor_id(),pipeline=SCHEMA,trainable=False)
    finally:cap.release()
