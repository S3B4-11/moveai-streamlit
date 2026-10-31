"""Redes compactas; ST-GCN simplificada con adyacencia fija de 33 landmarks."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score
from neural_features import EDGES,Normalizer,fingerprint

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(2)

def adjacency():
    a=np.eye(33,dtype=np.float32)
    for i,j in EDGES:a[i,j]=a[j,i]=1
    d=1/np.sqrt(a.sum(1));return torch.tensor(d[:,None]*a*d[None,:])

class Recurrent(nn.Module):
    def __init__(self,family,hidden=32,dropout=.2):
        super().__init__();self.project=nn.Sequential(nn.Linear(330,hidden),nn.ReLU())
        self.recurrent=(nn.GRU if family=="gru" else nn.LSTM)(hidden,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x):
        valid=(x[:,:,:,5]>=.55).any(-1).float()
        h,_=self.recurrent(self.project(x.flatten(2)))
        pooled=(h*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        return self.head(pooled).squeeze(-1)

class GraphBlock(nn.Module):
    def __init__(self,inc,outc,dropout):
        super().__init__();self.register_buffer("a",adjacency())
        self.spatial=nn.Conv2d(inc,outc,1)
        self.temporal=nn.Sequential(nn.GroupNorm(4,outc),nn.ReLU(),
            nn.Conv2d(outc,outc,(3,1),padding=(1,0)),nn.GroupNorm(4,outc),nn.Dropout(dropout))
        self.residual=nn.Conv2d(inc,outc,1) if inc!=outc else nn.Identity()
    def forward(self,x):
        graph=torch.einsum("vw,bctw->bctv",self.a,x)
        return torch.relu(self.temporal(self.spatial(graph))+self.residual(x))

class STGCN(nn.Module):
    def __init__(self,hidden=32,dropout=.2):
        super().__init__();self.blocks=nn.Sequential(GraphBlock(10,16,dropout),
            GraphBlock(16,hidden,dropout),GraphBlock(hidden,hidden,dropout));self.head=nn.Linear(hidden,1)
    def forward(self,x):
        mask=(x[:,:,:,5]>=.55).float()
        h=self.blocks(x.permute(0,3,1,2))
        pooled=(h*mask[:,None]).sum((2,3))/mask.sum((1,2)).clamp_min(1)[:,None]
        return self.head(pooled).squeeze(-1)

def build(family,config):
    if family not in ["gru","lstm","stgcn"]:raise ValueError(family)
    return STGCN(config["hidden"],config["dropout"]) if family=="stgcn" else Recurrent(family,config["hidden"],config["dropout"])

def logits(model,x,device="cpu",batch=32):
    model=model.to(device);model.eval();values=[]
    with torch.no_grad():
        for i in range(0,len(x),batch):values.append(model(torch.from_numpy(x[i:i+batch]).to(device)).cpu().numpy())
    return np.concatenate(values) if values else np.zeros(0)

def fit(family,config,x,mask,y,train,val=None,seed=42,max_epochs=60,patience=10,device="cpu",fixed_epochs=None):
    seed_all(seed);normalizer=Normalizer().fit(x[train],mask[train]);z=normalizer.transform(x,mask)
    model=build(family,config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=1e-3)
    yt=y[train];positive=max(int(yt.sum()),1);negative=max(len(yt)-positive,1)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/positive,device=device))
    loader=DataLoader(TensorDataset(torch.from_numpy(z[train]),torch.tensor(yt,dtype=torch.float32)),
        batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed))
    history=[];best=-float("inf");best_state=None;best_epoch=1;stale=0;started=time.perf_counter()
    for epoch in range(1,(fixed_epochs or max_epochs)+1):
        model.train();total=0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device);optimizer.zero_grad();loss=lossfn(model(xb),yb)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();total+=float(loss.detach())*len(yb)
        row={"epoch":epoch,"train_loss":total/len(train)}
        if val is not None:
            score=logits(model,z[val],device);auc=roc_auc_score(y[val],score)
            row["val_auc"]=float(auc)
            if auc>best+1e-5:
                best=auc;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
            else:stale+=1
        history.append(row)
        if val is not None and stale>=patience:break
    if best_state is not None:model.load_state_dict(best_state)
    model=model.cpu()
    return model,normalizer,{"best_epoch":best_epoch if val is not None else len(history),"history":history,
        "seconds":time.perf_counter()-started,"parameters":sum(p.numel() for p in model.parameters())}

def save_bundle(folder,model,family,config,normalizer,calibration,metadata=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    weight=folder/"weights.npz"
    np.savez_compressed(weight,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    meta=dict(family=family,config=config,normalizer=normalizer.as_dict(),calibration=calibration,
        representation=fingerprint(),weights_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),schema="moveai_neural_rgb_v1")
    meta.update(metadata or {});(folder/"model.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return meta

def load_bundle(folder):
    folder=Path(folder);meta=json.loads((folder/"model.json").read_text())
    if meta["schema"]!="moveai_neural_rgb_v1" or meta["representation"]!=fingerprint():raise ValueError("Representación incompatible.")
    if hashlib.sha256((folder/"weights.npz").read_bytes()).hexdigest()!=meta["weights_sha256"]:raise ValueError("Pesos alterados.")
    model=build(meta["family"],meta["config"])
    with np.load(folder/"weights.npz",allow_pickle=False) as data:
        model.load_state_dict({k:torch.from_numpy(data[k].copy()) for k in data.files})
    model.eval();return model,Normalizer(**meta["normalizer"]),meta

def probability(score,calibration):
    z=calibration["coefficient"]*np.asarray(score)+calibration["intercept"]
    return 1/(1+np.exp(-np.clip(z,-40,40)))
