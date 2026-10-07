"""Calidad de la extremidad activa y separación temporal de ciclos."""
import numpy as np
from geometry import JOINTS,signals,torso_size,q,smooth

def camera_view(image):
    r=q(np.linalg.norm(image[:,11]-image[:,12],axis=1)/np.maximum(torso_size(image),1e-6),50)
    return "perfil" if r is not None and r<.28 else "frente" if r is not None and r>.48 else "diagonal"

def track(sample,exercise,side):
    world,image,visibility=sample["world"],sample["image"],sample["visibility"]
    s,e,w,h,k,a=JOINTS[side]
    activity="knee" if exercise in ("squat","inline_lunge") else "arm" if exercise=="shoulder_abduction" else "elbow"
    joints=[h,k,a] if activity=="knee" else [s,e,h] if activity=="arm" else [s,e,w]
    sw,si=signals(world),signals(image);key=activity+"_"+side
    seen=(np.min(visibility[:,joints],axis=1)>=.55)&sample["detected"]
    valid3=seen&np.isfinite(world[:,joints]).all(axis=(1,2))
    bones=[(h,k),(k,a)] if activity=="knee" else [(s,e)] if activity=="arm" else [(s,e),(e,w)]
    projected=np.ones(len(world),bool);variation=[]
    for i,j in bones:
        length=np.linalg.norm(world[:,i]-world[:,j],axis=1);median=q(length[valid3],50)
        if median:
            valid3 &= (length>.55*median)&(length<1.6*median)
            variation.append(float(np.nanstd(length[valid3])/median) if valid3.any() else 1.)
        else:valid3[:]=False
        length2=np.linalg.norm(image[:,i]-image[:,j],axis=1);reference=q(length2[seen],90)
        projected &= length2/np.maximum(torso_size(image),1e-6)>=.18
        if reference:projected &= length2>=.6*reference
    valid3 &= ~np.r_[False,abs(np.diff(sw[key]))>55]
    valid2=seen&projected&np.isfinite(si[key])
    valid2 &= ~np.r_[False,abs(np.diff(si[key]))>55]
    view=camera_view(image)
    plane_ok=activity=="elbow" or (activity=="arm" and view=="frente") or (activity=="knee" and view!="frente")
    use2=bool(plane_ok and valid2.mean()>=.5 and valid2.sum()>=8)
    chosen,mask,source=(si[key],valid2,"2D observable") if use2 else (sw[key],valid3,"3D estimada")
    reason=""
    disagreement=q(abs(sw[key][valid2]-si[key][valid2]),95) if valid2.any() else None
    if not use2 and disagreement is not None and disagreement>45:
        reason="Las estimaciones del recorrido difieren mucho; cambia la toma para medir esta articulación."
    if not use2 and max(variation,default=1.)>.25:
        reason="Los segmentos estimados cambian demasiado de longitud; revisa la toma y las oclusiones."
    trunk=(image[:,11]+image[:,12]-image[:,23]-image[:,24])/2
    trunk_visible=(np.min(visibility[:,[11,12,23,24]],axis=1)>=.55)&sample["detected"]
    vertical_fraction=q(abs(trunk[trunk_visible,1])/np.maximum(np.linalg.norm(trunk[trunk_visible],axis=1),1e-6),50)
    if vertical_fraction is not None and vertical_fraction<.45:
        reason="La pose del torso no ofrece una referencia fiable para esta toma. Revisa el encuadre."
    if seen.mean()<.8:
        reason="La extremidad activa queda oculta en parte del video. Grábala visible durante todo el recorrido."
    radius=max(1,round(.12/max(float(np.median(np.diff(sample["times"]))),1e-3)))
    measured=smooth(chosen,mask,radius)
    # La pose puede localizar un retorno cuando la proyección se acorta. No
    # sustituye los ángulos medidos ni rellena puntos con baja visibilidad.
    cycles=np.where(mask,chosen,np.where(valid3,sw[key],np.nan)) if use2 else np.where(mask,chosen,np.nan)
    cycles=smooth(cycles,np.isfinite(cycles),radius)
    wrist=q(image[:,w,0],50);mid=q((image[:,11,0]+image[:,12,0])/2,50)
    screen="izquierda de la imagen" if wrist is not None and mid is not None and wrist<mid else "derecha de la imagen"
    return dict(side=side,screen_side=screen,series=measured,cycle_series=cycles,valid=mask,joints=joints,
                source=source,range=(q(measured,95) or 0)-(q(measured,5) or 0),
                visibility=float(mask.mean()),pose_visibility=float(seen.mean()),view_observed=view,
                reason=reason,disagreement=disagreement,bone_variation=max(variation,default=1.))

def repetitions(series,times):
    series=np.asarray(series);valid=np.isfinite(series)
    if valid.sum()<8:return []
    low,high=np.percentile(series[valid],[10,95]);amplitude=high-low
    if amplitude<20:return []
    enter,leave=low+.55*amplitude,low+.25*amplitude
    fps=1/max(float(np.median(np.diff(times))),1e-3);padding=max(1,int(.18*fps))
    result=[];active=False;last_low=None;missing=0;start=peak=0;complete=False
    for i,value in enumerate(series):
        if not np.isfinite(value):
            missing+=1
            if missing>max(2,int(.35*fps)):
                if active:result.append(dict(start=start,peak=peak,end=i,complete=False,reason="Oclusión durante la repetición."))
                active=False;last_low=None
            continue
        missing=0
        if not active:
            if value<=leave:last_low=i
            if value>=enter:
                start=max(0,(last_low if last_low is not None else i)-padding)
                complete=last_low is not None;peak=i;active=True
        else:
            if value>series[peak]:peak=i
            if value<=leave and times[i]-times[start]>=.5:
                result.append(dict(start=start,peak=peak,end=min(len(series),i+padding+1),
                                   complete=bool(complete),reason="" if complete else "Falta el inicio completo."))
                active=False;last_low=i
    if active:result.append(dict(start=start,peak=peak,end=len(series),complete=False,reason="Falta observar el retorno."))
    return result

def active_tracks(sample,exercise,side="automatica"):
    tracks=[track(sample,exercise,s) for s in JOINTS]
    if side!="automatica":return [t for t in tracks if t["side"]==side]
    if exercise in ("squat","inline_lunge"):return [max(tracks,key=lambda t:t["range"]*t["visibility"])]
    best=max(t["range"]*t["visibility"] for t in tracks)
    return [t for t in tracks if t["range"]>=20 and t["range"]*t["visibility"]>=max(12,.3*best)]
