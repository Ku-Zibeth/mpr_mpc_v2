"""Deterministic initial-state evidence for paired evaluation."""
from pathlib import Path
import hashlib,json,random
import numpy as np
import torch

def clean(x):
    if torch.is_tensor(x):return clean(x.detach().cpu().numpy())
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,Path):return str(x)
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def dump(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean(x),indent=2,allow_nan=False))
def load(path):return json.loads(Path(path).read_text())
def digest(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rng_state(obj):
    rng=getattr(obj,'np_random',None) if obj is not None else None
    if rng is None:return None
    return clean(rng.get_state() if hasattr(rng,'get_state') else rng.bit_generator.state)


def snapshot(env,obs):
    sim=env.unwrapped_metadrive;v=sim.agent;traffic=[]
    for obj in sim.engine.get_objects().values():
        if obj is v or not hasattr(obj,'velocity') or not hasattr(obj,'position'):continue
        traffic.append([type(obj).__name__,*map(float,obj.position[:2]),*map(float,obj.velocity[:2]),float(getattr(obj,'heading_theta',0.))])
    vehicle=dict(position=v.position,velocity=v.velocity,heading=float(v.heading_theta))
    for name in ['steering','throttle_brake','last_current_action','last_position','last_heading_dir','current_action','last_steering']:
        value=getattr(v,name,None)
        if value is not None and isinstance(value,(float,int,list,tuple,np.ndarray,np.generic,dict)):vehicle[name]=clean(value)
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.random.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(),
        engine=rng_state(sim.engine),traffic_manager=rng_state(getattr(sim.engine,'traffic_manager',None)))
    values=dict(observation=clean(obs),vehicle=clean(vehicle),traffic=sorted(traffic),rng=clean(rng))
    return values|dict(hashes={k:digest(v) for k,v in values.items()},rng_component_hashes={k:digest(v) for k,v in rng.items()})

