"""Seguimiento optativo de una persona señalada, sin alterar el extractor entrenado."""
from pathlib import Path
import hashlib
import numpy as np
from pose import FPS, asset, extractor_id
from geometry import SCHEMA

def validate_target(target):
    p=np.asarray(target,dtype=float)
    if p.shape!=(2,) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError('Selecciona una posición dentro del video.')
    return p

def select_person(poses,target,anchor=None):
    """Centro del torso y continuidad; una coincidencia ambigua se rechaza."""
    target=validate_target(target);reference=target if anchor is None else np.asarray(anchor)
    candidates=[]
    for i,pose in enumerate(poses):
        torso=pose[11:13]+pose[23:25]
        if min(min(p.visibility or 0,p.presence if p.presence is not None else 1) for p in torso)<.4:continue
        center=np.mean([[p.x,p.y] for p in torso],axis=0)
        if not np.isfinite(center).all():continue
        distance=float(np.linalg.norm(center-reference))
        if distance<= (.28 if anchor is None else .18):candidates.append((distance,i,center))
    candidates.sort(key=lambda x:x[0])
    if not candidates:return None,False,None
    if len(candidates)>1 and candidates[1][0]-candidates[0][0]<.055:
        return None,True,None
    _,index,center=candidates[0]
    return index,False,center

def selected_extractor_id():
    return hashlib.sha256((extractor_id()+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()

def extract_selected_video(path,target,previews=True):
    import cv2
    import mediapipe as mp
    target=validate_target(target);cv2.setNumThreads(1);cap=cv2.VideoCapture(str(path))
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not np.isfinite(fps) or fps<=0 or frames<8:raise ValueError('No se pudo leer una secuencia suficiente.')
        if frames/fps>60.5:raise ValueError('Recorta el video a un máximo de 60 segundos.')
        stride=max(1,int(np.ceil(fps/FPS)))
        opts=mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(asset()),delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=4)
        world=[];image=[];visibility=[];times=[];detected=[];ambiguous=[];thumbs=[]
        anchor=None;last_seen=None;initial_center=None;index=0;last_ms=-1
        with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as detector:
            while cap.grab():
                if index%stride==0:
                    ok,frame=cap.retrieve()
                    if not ok:break
                    time=index/fps;h,w=frame.shape[:2];scale=min(1.,720/max(h,w))
                    rgb=cv2.cvtColor(cv2.resize(frame,(round(w*scale),round(h*scale))),cv2.COLOR_BGR2RGB)
                    value=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    ms=max(last_ms+1,round(time*1000));last_ms=ms
                    poses=detector.detect_for_video(value,ms)
                    # Después de perder el seguimiento no se cambia de persona.
                    # Sólo se admite la misma zona inicial; un corte de cámara
                    # incompatible deja de ser evaluable.
                    reference=anchor
                    if last_seen is not None and time-last_seen>1.5:reference=initial_center
                    person,multi,center=select_person(poses.pose_landmarks,target,reference)
                    present=person is not None
                    cw=np.full((33,3),np.nan,np.float32);ci=np.full((33,2),np.nan,np.float32);v=np.zeros(33,np.float32)
                    if present:
                        if initial_center is None:initial_center=center.copy()
                        anchor=center.copy();last_seen=time
                        cw=np.array([[p.x,p.y,p.z] for p in poses.pose_world_landmarks[person]],np.float32)
                        ci=np.array([[p.x*w/h,p.y] for p in poses.pose_landmarks[person]],np.float32)
                        v=np.array([min(p.visibility or 0,p.presence if p.presence is not None else 1) for p in poses.pose_landmarks[person]],np.float32)
                    world.append(cw);image.append(ci);visibility.append(v);times.append(time);detected.append(present);ambiguous.append(multi)
                    if previews and len(world)%3==1:
                        jpg=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,75])[1]
                        thumbs.append((len(world)-1,jpg.tobytes()))
                index+=1
        if len(world)<8:raise ValueError('No se leyó una secuencia suficiente.')
        world=np.asarray(world);good=np.asarray(detected)
        hip=(world[:,23]+world[:,24])/2;torso=np.linalg.norm((world[:,11]+world[:,12])/2-hip,axis=1)
        normalizer=float(np.nanmedian(torso[good])) if good.any() else 1.
        if not np.isfinite(normalizer) or normalizer<.01:normalizer=1.
        return dict(world=(world-hip[:,None,:])/normalizer,image=np.asarray(image),visibility=np.asarray(visibility),
                    times=np.asarray(times),detected=good,ambiguous=np.asarray(ambiguous),coverage=float(good.mean()),
                    duration=frames/fps,previews=thumbs,pose_version=mp.__version__,extractor_id=selected_extractor_id(),
                    pipeline=SCHEMA,trainable=False,person_selection=dict(target=target.tolist(),mode='explicit'))
    finally:cap.release()

def selection_preview(path,target):
    import cv2
    target=validate_target(target);cap=cv2.VideoCapture(str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_MSEC,1000*min(3.,frames/max(fps,1)/4))
        ok,frame=cap.read()
        if not ok:return None
        h,w=frame.shape[:2];scale=min(1.,640/w);frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
        h,w=frame.shape[:2];point=tuple(np.rint(target*[w,h]).astype(int))
        cv2.drawMarker(frame,point,(0,215,255),cv2.MARKER_CROSS,30,3)
        return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    finally:cap.release()
