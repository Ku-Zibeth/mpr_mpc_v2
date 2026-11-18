"""Isolated counterfactual protocol utilities; never part of online action selection."""
from __future__ import annotations
import contextlib,csv,hashlib,io,itertools,json,random,sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2')
REPORT=ROOT/'reports/mppi_first_action_value_20261117'
NATIVE=ROOT/'reports/mppi_native_parity_20261117'

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
def file_sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):d.update(b)
    return d.hexdigest()
def table(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(clean(rows))
def tensor_hash(model):
    d=hashlib.sha256()
    for n,t in sorted(model.state_dict().items()):
        if torch.is_tensor(t):d.update(n.encode());d.update(t.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()
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


def assert_pair(actual,expected):
    for key in ['observation','vehicle','traffic','rng']:
        if actual['hashes'][key]!=expected['hashes'][key]:raise ValueError('origin mismatch: '+key)


def physical_first_action(sequence):
    a=np.asarray(sequence,np.float32)
    if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all() or np.any(np.abs(a)>1):raise ValueError('Physical sequence must be finite [H+1,2] in[-1,1].')
    return a[0].copy()


def feature_vectors(latent,state535,refined_parameters,baseline_action,candidate_action):
    z,s,p,b,a=map(lambda x:np.asarray(x,np.float32).reshape(-1),(latent,state535,refined_parameters,baseline_action,candidate_action))
    if [len(x) for x in [z,s,p,b,a]]!=[512,535,3,2,2] or not np.array_equal(z,s[:512]):raise ValueError('Original online feature schema mismatch')
    return dict(F1=np.r_[z,a],F2=np.r_[s,a],F3=np.r_[s,p,b,a])


def make_runtime(scene):
    from mpr_mpc._bootstrap import bootstrap
    bootstrap()
    from common.seed import set_seed
    from env import make_env
    import evaluate_model_suite as suite
    p=load(REPORT/'protocol.json');ck=Path(p['checkpoint']);raw=OmegaConf.load(p['source_environment_config'])
    raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
    raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8
    _,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[scene]);torch.set_num_threads(1);torch.set_float32_matmul_precision('high');set_seed(1);env=make_env(cfg)
    from planning.sac_value import policy_context_sha256
    assert policy_context_sha256(cfg)==load(SOURCE/'reports/sac_value_20261117/dataset_manifest.json')['context'], 'FrozenBaseline simulator/reward/Lattice/residualcontext changed'
    with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);state=agent.load(ck)
    agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',state);agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
    assert agent.planner.allow_residual and not agent.planner.allow_mppi and agent.planner.residual_state_dim==535
    return cfg,env,agent


@torch.no_grad()
def at_origin(env,agent,scene,before):
    from common.seed import set_seed
    seed=scene+1;set_seed(seed);agent.planner.reset(seed=seed);obs=env.reset(seed=scene)
    for i in range(before-1):
        a=agent.act(obs,t0=i==0,eval_mode=True);obs,_,done,info=env.step(a)
        if done:return obs,dict(unreachable=True,steps=i+1,end_reason=info.get('end_reason',''))
    agent.act(obs,t0=False,eval_mode=True)
    assert agent.last_plan.stage.use_residual and not agent.last_plan.stage.use_mppi
    return obs,None


def all_reducers(q):
    q=np.asarray(q);out={}
    for a,b in itertools.combinations(range(5),2):out[f'avg{a}{b}']=q[:,[a,b]].mean(1);out[f'min{a}{b}']=q[:,[a,b]].min(1)
    return out|dict(mean=q.mean(1),median=np.median(q,axis=1),minimum=q.min(1))


def check_value_context(actual,expected):
    for k in ['wm_checkpoint_sha','sac_checkpoint_sha','training_manifest_sha','feature_schema','gamma','reward_definition','baseline_policy_sha','intervention_horizon','continuation_policy']:
        if k not in actual or actual[k]!=expected[k]:raise ValueError('value context mismatch: '+k)


def assert_feedback_stage(stage):
    if not stage.use_residual or stage.use_mppi:raise ValueError('Continuation must be residualON,MPPioff')


@torch.no_grad()
def terminal_policy_actions(model,z,candidate,seed):
    device=z.device;devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed);sample,info=model.pi(z,None)
    return dict(candidate=candidate,policy_mean=info['mean'],native_policy_sample=sample)


def representative_candidates(pool,values,selected_index,seed):
    """Fixed predicted-score quantiles; deduplicate EXACT physical first actions."""
    pool,values=np.asarray(pool,np.float32),np.asarray(values)
    order=np.argsort(values[1:],kind='stable')+1
    roles=[('baseline',0),('mppi_selected',selected_index),('best_nonbaseline',int(order[-1]))]
    roles.extend((name,int(order[int(round(q*(len(order)-1)))])) for name,q in [('q75',.75),('q50',.5),('q25',.25)])
    roles.extend([('worst_nonbaseline',int(order[0])),('fixed_gaussian',int(np.random.default_rng(seed).integers(2,len(pool))))])
    representatives=[];aliases=[];seen={}
    for role,index in roles:
        first=physical_first_action(pool[index]);key=first.tobytes()
        if key not in seen:
            seen[key]=len(representatives);representatives.append(dict(candidate_id=f'c{len(representatives)}',pool_index=index,roles=[role]))
        else:
            prior=representatives[seen[key]];prior['roles'].append(role);aliases.append(dict(role=role,original_pool_index=index,representative=prior['candidate_id'],different_tail=not np.array_equal(pool[index],pool[prior['pool_index']])))
    return representatives,aliases
