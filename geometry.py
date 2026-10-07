"""Ángulos observables de videos RGB; cumplimiento no equivale a probabilidad."""
import numpy as np

SCHEMA = "moveai_rgb_v4"
EXERCISES = {"elbow_flexion":"Curl de bíceps", "squat":"Sentadilla",
             "shoulder_abduction":"Elevación lateral", "inline_lunge":"Zancada"}
TARGETS = {"elbow_flexion":120, "squat":80, "shoulder_abduction":90, "inline_lunge":85}
JOINTS = {"izquierda":(11,13,15,23,25,27), "derecha":(12,14,16,24,26,28)}

def q(values, percentile):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return float(np.percentile(a,percentile)) if len(a) else None

def angle(a,b,c):
    u,v=np.asarray(a)-b,np.asarray(c)-b
    d=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
    value=np.degrees(np.arccos(np.clip(np.sum(u*v,axis=-1)/np.maximum(d,1e-8),-1,1)))
    return np.where(d>1e-6,value,np.nan)

def signals(coords):
    c=np.asarray(coords,float);out={}
    for side,(s,e,w,h,k,a) in JOINTS.items():
        out["elbow_"+side]=180-angle(c[:,s],c[:,e],c[:,w])
        out["knee_"+side]=180-angle(c[:,h],c[:,k],c[:,a])
        out["arm_"+side]=angle(c[:,e],c[:,s],c[:,h])
        out["hip_"+side]=180-angle(c[:,s],c[:,h],c[:,k])
    return out

def torso_size(c):
    c=np.asarray(c,float)
    return np.linalg.norm((c[:,11]+c[:,12]-c[:,23]-c[:,24])/2,axis=1)

def smooth(values,valid,radius=1):
    a=np.asarray(values,float).copy();a[~valid]=np.nan;out=np.full_like(a,np.nan)
    for i in range(len(a)):
        if np.isfinite(a[i]):
            nearby=a[max(0,i-radius):min(len(a),i+radius+1)]
            nearby=nearby[np.isfinite(nearby)]
            if len(nearby):out[i]=np.median(nearby)
    return out

def trunk_change(c,start,end,valid=None):
    c=np.asarray(c,float);v=(c[:,11]+c[:,12]-c[:,23]-c[:,24])/2
    length=np.linalg.norm(v,axis=1);good=np.isfinite(v).all(axis=1)
    if valid is not None:good &= np.asarray(valid,bool)
    reference_length=q(length[good],50)
    if reference_length:good &= (length>=.6*reference_length)&(length<=1.6*reference_length)
    v=v.copy();v[~good]=np.nan
    # La referencia inicial sólo usa articulaciones visibles y un torso de
    # longitud plausible; una pose inventada detrás de una oclusión no la fija.
    usable=v[good][:6]
    if not len(usable):return np.full(end-start,np.nan)
    ref=np.median(usable,axis=0);v=v[start:end]
    d=np.linalg.norm(v,axis=1)*np.linalg.norm(ref)
    return np.degrees(np.arccos(np.clip(v@ref/np.maximum(d,1e-8),-1,1)))

def criterion(key,name,value,reference,direction="min",tolerance=10,weight=1,advice=""):
    margin=None if value is None else value-reference if direction=="min" else reference-value
    state="no_evaluable" if margin is None else "en_rango" if margin>=0 else "limite" if margin>=-tolerance else "fuera_de_rango"
    return dict(key=key,name=name,value=None if value is None else round(value,1),
                reference=float(reference),direction=direction,status=state,weight=weight,advice=advice)
