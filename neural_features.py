"""Representación RGB temporal compartida por GRU, LSTM y ST-GCN.

No depende de MediaPipe ni de PyTorch. Las ausencias tienen máscara explícita.
"""
import hashlib
from pathlib import Path
import numpy as np
from geometry import signals

STEPS = 64
CHANNELS = ["world_x", "world_y", "world_z", "image_x", "image_y",
            "visibility", "angle_world/180", "angle_image/180", "valid_world_angle", "valid_image_angle"]
ANGLES = [(11,"arm_izquierda",(13,11,23)),(12,"arm_derecha",(14,12,24)),
          (13,"elbow_izquierda",(11,13,15)),(14,"elbow_derecha",(12,14,16)),
          (23,"hip_izquierda",(11,23,25)),(24,"hip_derecha",(12,24,26)),
          (25,"knee_izquierda",(23,25,27)),(26,"knee_derecha",(24,26,28))]
EDGES = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
         (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
         (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),
         (23,24),(23,25),(24,26),(25,27),(26,28),(27,29),(28,30),
         (29,31),(30,32),(27,31),(28,32)]

def fingerprint():
    root=Path(__file__).parent
    return hashlib.sha256(b"".join((root/n).read_bytes() for n in
        ["neural_features.py","neural_models.py","geometry.py"])).hexdigest()

def encode(sample,steps=STEPS):
    t=np.asarray(sample["times"],float)
    if len(t)<2 or not np.all(np.diff(t)>0): raise ValueError("Tiempos insuficientes o no crecientes.")
    grid=np.linspace(t[0],t[-1],steps)
    # Vecino más cercano: no inventa articulaciones durante oclusiones.
    right=np.clip(np.searchsorted(t,grid),1,len(t)-1);left=right-1
    indices=np.where(abs(t[right]-grid)<abs(t[left]-grid),right,left)
    world=np.asarray(sample["world"])[indices];image=np.asarray(sample["image"])[indices]
    vis=np.clip(np.nan_to_num(np.asarray(sample["visibility"])[indices]),0,1)
    frame=np.asarray(sample["detected"])[indices]&~np.asarray(sample["ambiguous"])[indices]
    seen=(vis>=.55)&frame[:,None]
    x=np.zeros((steps,33,10),np.float32);mask=np.zeros_like(x,bool)
    x[:,:,:3]=world;x[:,:,3:5]=image;x[:,:,5]=vis*frame[:,None]
    mask[:,:,:3]=seen[:,:,None]&np.isfinite(world)
    mask[:,:,3:5]=seen[:,:,None]&np.isfinite(image);mask[:,:,5]=True
    for space,coords,channel,flag in [("world",world,6,8),("image",image,7,9)]:
        sig=signals(coords)
        for node,key,joints in ANGLES:
            valid=seen[:,joints].all(axis=1)&np.isfinite(coords[:,joints]).all(axis=(1,2))&np.isfinite(sig[key])
            x[:,node,channel]=sig[key]/180;mask[:,node,channel]=valid
            x[:,node,flag]=valid;mask[:,node,flag]=True
    x[~mask]=0
    if not np.isfinite(x).all():raise ValueError("Tensor no finito.")
    return x,mask

class Normalizer:
    def __init__(self,mean=None,scale=None):
        self.mean=np.zeros(10,np.float32) if mean is None else np.asarray(mean,np.float32)
        self.scale=np.ones(10,np.float32) if scale is None else np.asarray(scale,np.float32)
    def fit(self,x,mask):
        for c in [0,1,2,3,4,6,7]:
            values=x[...,c][mask[...,c]]
            if len(values):self.mean[c]=values.mean();self.scale[c]=max(values.std(),1e-4)
        return self
    def transform(self,x,mask):
        result=(np.asarray(x,np.float32)-self.mean)/self.scale
        result[~mask]=0
        return result.astype(np.float32)
    def as_dict(self):return {"mean":self.mean.tolist(),"scale":self.scale.tolist()}
