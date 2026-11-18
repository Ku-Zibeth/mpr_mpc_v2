"""Small, baseline-centered offline advantage models, never online imports."""
import numpy as np
import torch
from torch import nn


SCHEMAS={'A1':537,'A2':539}


def features(state,baseline,candidate,schema):
    s,b,a=[np.asarray(x,np.float32).reshape(-1) for x in (state,baseline,candidate)]
    if s.shape!=(535,) or b.shape!=(2,) or a.shape!=(2,):raise ValueError('Current535 and physical commands required')
    if not all(np.isfinite(x).all() for x in (s,b,a)):raise ValueError('Nonfinite current inputs')
    if schema=='A0':return np.r_[s,a]
    if schema=='A1':return np.r_[s,a-b]
    if schema=='A2':return np.r_[s,b,a-b]
    raise ValueError('Unknown fixed schema')


def relative_targets(y,origin,baseline):
    y=np.asarray(y,float);out=np.empty_like(y)
    for o in sorted(set(origin)):
        ids=np.flatnonzero(np.asarray(origin)==o)
        refs=[i for i in ids if baseline[i]]
        if len(refs)!=1:raise ValueError('Require ONE actual baseline within each origin')
        out[ids]=y[ids]-y[refs[0]]
    return out


def balanced_weights(scene,origin):
    scene,origin=np.asarray(scene),np.asarray(origin)
    scenes=sorted(set(scene));w=np.empty(len(scene),np.float32)
    for s in scenes:
        os=sorted(set(origin[scene==s]))
        for o in os:
            ids=(scene==s)&(origin==o)
            w[ids]=1/len(scenes)/len(os)/int(ids.sum())
    return w


def within_origin_pairs(scene,origin,delta_y):
    scene,origin,y=np.asarray(scene),np.asarray(origin),np.asarray(delta_y)
    groups=[]
    for o in sorted(set(origin)):
        ids=np.flatnonzero(origin==o)
        if len(set(scene[ids]))!=1:raise ValueError('Origin cannot span maps')
        pairs=[(i,j) for n,i in enumerate(ids) for j in ids[n+1:] if abs(y[i]-y[j])>1.]
        if pairs:groups.append((scene[ids[0]],o,pairs))
    maps=set(s for s,_,_ in groups);pairs=[];weights=[]
    for s,o,ps in groups:
        n=sum(other==s for other,_,_ in groups)
        for i,j in ps:pairs.append((i,j));weights.append(1/len(maps)/n/len(ps))
    return np.asarray(pairs,np.int64).reshape(-1,2),np.asarray(weights,np.float32)


def fit_stats(x,y,weights):
    x,y,w=map(torch.as_tensor,(x,y,weights))
    x,y,w=x.float(),y.float(),w.float()
    mu=(w[:,None]*x).sum(0)
    scale=(w[:,None]*(x-mu).square()).sum(0).sqrt().clamp_min(.001)
    # Scale only, no mean shift: true Baseline advantage must remain ZERO.
    target_scale=(w*y.square()).sum().sqrt().clamp_min(1.)
    return mu,scale,target_scale


class Advantage(nn.Module):
    def __init__(self,dimension,kind):
        super().__init__();self.dimension=dimension;self.kind=kind
        self.register_buffer('mu',torch.zeros(dimension));self.register_buffer('scale',torch.ones(dimension))
        self.register_buffer('target_scale',torch.ones(()))
        self.net=nn.Linear(dimension,1) if kind=='linear' else nn.Sequential(nn.Linear(dimension,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
        if kind=='linear':
            nn.init.zeros_(self.net.weight);nn.init.zeros_(self.net.bias)
    def raw(self,x):return self.net((x-self.mu)/self.scale).squeeze(-1)
    def forward(self,x):
        if x.shape[-1]!=self.dimension:raise ValueError('Fixed feature schema mismatch')
        zero=x.clone();zero[:,-2:]=0
        return (self.raw(x)-self.raw(zero))*self.target_scale


class LocalKNN(nn.Module):
    """Diagnostic A2 k=8 control; same features, no learned neural parameters."""
    def __init__(self,x,y,w,k=8):
        super().__init__();self.dimension=x.shape[1];self.kind='knn';self.k=min(k,len(x))
        mu,scale,target=fit_stats(x,y,w)
        self.register_buffer('mu',mu);self.register_buffer('scale',scale);self.register_buffer('target_scale',target)
        self.register_buffer('block',torch.tensor([535**-.5]*535+[2**-.5]*4))
        self.register_buffer('bank',(torch.as_tensor(x).float()-mu)/scale*self.block)
        self.register_buffer('labels',torch.as_tensor(y).float()/target)
        self.register_buffer('weights',torch.as_tensor(w).float())
    def raw(self,x):
        normalized=(x-self.mu)/self.scale*self.block
        distance=torch.cdist(normalized,self.bank)
        d,ids=torch.topk(distance,self.k,dim=1,largest=False,sorted=True)
        w=self.weights[ids]/d.clamp_min(1e-6)
        return (w*self.labels[ids]).sum(1)/w.sum(1)
    def forward(self,x):
        zero=x.clone();zero[:,-2:]=0
        return (self.raw(x)-self.raw(zero))*self.target_scale


def holdout(data,scene):
    train=np.flatnonzero(data['scene']!=scene);test=np.flatnonzero(data['scene']==scene)
    if not len(train) or not len(test):raise ValueError('LOMO requires train and test maps')
    return train,test
