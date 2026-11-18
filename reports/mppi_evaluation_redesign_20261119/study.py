"""H1-only experiment utilities. Never imported by an online controller."""
import contextlib,copy,csv,hashlib,io,json,random,sys,os
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
ROOT=R.parents[1]
MAIN=ROOT.parent/'mpr_mpc_v2'
HIST=ROOT.parent/'mpr_mpc_v2_h1_counterfactual_20261118'/'reports/mppi_h1_counterfactual_20261118'
SOURCE=R
CODE_ROOT=Path(os.environ.get('MPR_CODE_ROOT',str(MAIN)))
sys.path.insert(0,str(CODE_ROOT))
from tools.first_action_study import clean,digest,snapshot,tensor_hash,file_sha,rng_state

def load(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(clean(x),f,indent=2,allow_nan=False)
def table(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(clean(rows))
def origin_dir(scene,seed):return R/'origins'/f'{scene}_r{seed}'
def original_dir(scene,seed):return R/'baselines'/f'{scene}_r{seed}'

def numeric_state(obj,depth=1):
    """Capture scalar/array/PID history, excluding engine/object references and process addresses."""
    result={}
    for name,value in vars(obj).items():
        if isinstance(value,(str,bool,int,float,np.generic,np.ndarray)) or value is None:result[name]=clean(value)
        elif isinstance(value,(list,tuple)) and all(isinstance(x,(str,bool,int,float,np.generic,np.ndarray)) or x is None for x in value):result[name]=clean(value)
        elif depth and 'pid' in name.lower() and hasattr(value,'__dict__'):result[name]=numeric_state(value,depth-1)
    return result

def extended_environment(env):
    sim=env.unwrapped_metadrive;engine=sim.engine
    managers={name:rng_state(value) for name,value in engine.managers.items()}
    objects=[]
    for obj in engine.get_objects().values():
        if not hasattr(obj,'position'):continue
        value=dict(type=type(obj).__name__,position=clean(obj.position),rng=rng_state(obj))
        for name in ['velocity','heading_theta','angular_velocity','steering','throttle_brake','current_action','crash_vehicle','crash_object','out_of_road']:
            if hasattr(obj,name):value[name]=clean(getattr(obj,name))
        policy=engine.get_policy(obj.id)
        if policy is not None:value['policy']=dict(type=type(policy).__name__,rng=rng_state(policy),history=numeric_state(policy))
        objects.append(value)
    # Object IDs can be UUIDs; canonicalize by physical identity, not process addresses.
    objects.sort(key=lambda x:(x['type'],str(x['position'])))
    # Gymnasium's np_random getter CREATES an entropy-seeded generator when
    # _np_random is None. Observing an uninitialized, unused RNG must not create it.
    gym_rng=getattr(sim,'_np_random',None)
    env_rng=None if gym_rng is None else clean(gym_rng.get_state() if hasattr(gym_rng,'get_state') else gym_rng.bit_generator.state)
    values=dict(environment_rng=env_rng,environment_rng_initialized=gym_rng is not None,
        managers=managers,objects=objects,episode_step=int(sim.episode_step))
    return dict(values=values,hash=digest(values))

def get_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.random.get_rng_state().clone(),cuda=torch.cuda.get_rng_state().clone())
def set_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.random.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'])

def make_runtime(scene,seed=1,mode='baseline',record=True):
    from mpr_mpc._bootstrap import bootstrap
    bootstrap()
    from omegaconf import OmegaConf
    from common.seed import set_seed
    from env import make_env
    import evaluate_model_suite as suite
    p=load(R/'protocol.json');ck=Path(p['checkpoint'])
    raw=OmegaConf.load(p['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(CODE_ROOT/'config.yaml').mpr_mpc.mppi
    raw.mpr_mpc.mppi.record_candidates=bool(record)
    if mode=='modified':
        choice=load(R/'decision.json')['online_options']
        for k,v in choice.items():raw.mpr_mpc.mppi[k]=v
    raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False
    raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8;raw.seed=seed
    parsed,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[scene])
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');set_seed(1)
    env=make_env(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        agent=suite._make_agent(cfg,env);state=agent.load(ck)
    agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl' if mode=='baseline' else 'full_mpr_mpc',state)
    agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
    assert agent.planner.residual_state_dim==535 and agent.planner.allow_mppi==(mode!='baseline')
    assert float(agent.tdmpc_agent.discount)==.995 and agent.planner.control_dt==.1
    assert agent.planner.local_mppi.num_samples==64 and agent.planner.local_mppi.num_elites==8 and agent.planner.local_mppi.iterations==4
    assert not agent.planner.local_mppi.compare_final_elite and agent.planner.local_mppi.cost_weight==0
    return parsed,cfg,env,agent

def policy_snapshot(agent):
    p=agent.planner;c=p.last_context;result=agent.last_plan;l=p.structured.planner
    values=dict(latent=c.latent,residual535=c.residual_state,coarse_actions=c.coarse_actions,
        refined_actions=result.refined_actions,refined_parameters=result.refined_parameters,
        requested_residual=result.requested_normalized_residual,
        native_td_prev_mean=agent.tdmpc_agent._prev_mean,
        global_step=agent.global_env_step,pending_residual=agent._pending_residual,
        local_generator=p.local_mppi.generator.get_state(),episode_seed=p._episode_seed,
        lattice=dict(reference=l.last_reference_points,f_state=l.last_f_state,lane_width=l.last_lane_width,
            lateral_bounds=l.last_lateral_bounds,lane_centers=l.last_lane_centers,selected_index=l.last_selected_index,
            candidates=[dict(xy=x.xy,t=x.t,s_d=x.s_d,d=x.target_d,v=x.target_speed,h=x.horizon) for x in l.last_candidates]),
        safety_previous_action=c.safety_context.previous_action if c.safety_context is not None else None,
        risk_status=p.residual_state_builder.last_risk_status)
    assert agent._pending_residual is None
    return dict(values=clean(values),hashes={k:digest(v) for k,v in values.items()},
        semantics='Feed-forward encoder and SAC; native TD previous mean dormant. Last plan diagnostics overwritten, not action history input.')

@torch.no_grad()
def restore_origin(env,agent,scene,seed):
    """Replay exactly step-1 recorded controls, rebuilding policy on every prior tick."""
    from common.seed import set_seed
    old=original_dir(scene,seed);trace=np.load(old/'trace.npz');step=load(R/'protocol.json')['origin_step']
    assert len(trace['actions'])>=step
    combined=scene+seed;set_seed(combined);agent.planner.reset(seed=combined);obs=env.reset(seed=scene)
    assert np.array_equal(obs.cpu().numpy(),trace['observations'][0])
    for i in range(step-1):
        actual=agent.act(obs,t0=i==0,eval_mode=True).cpu().numpy()
        assert np.array_equal(actual,trace['actions'][i]),('prefix action',scene,step,i)
        obs,reward,done,info=env.step(trace['actions'][i])
        assert not done and abs(float(info['tdmpc2_reward'])-float(trace['rewards'][i]))<1e-8
        assert np.array_equal(obs.cpu().numpy(),trace['observations'][i+1]),('prefix observation',scene,step,i)
    # Historical snapshots were captured AFTER decision computation but BEFORE env.step.
    before_decision_rng=get_rng()
    action=agent.act(obs,t0=False,eval_mode=True)
    assert np.array_equal(action.cpu().numpy(),trace['actions'][step-1])
    current=snapshot(env,obs);expected=load(old/'origin.json')
    assert current['hashes']==expected['hashes'],('origin',scene,step,{k:(current['hashes'][k],expected['hashes'][k]) for k in current['hashes'] if current['hashes'][k]!=expected['hashes'][k]})
    with np.load(old/'origin.npz') as saved:
        assert np.array_equal(agent.last_plan.refined_actions.cpu().numpy(),saved['refined_actions'])
    policy=policy_snapshot(agent)
    return obs,action,trace,current,policy,before_decision_rng

def representative_roles(records,epsilon):
    """Keep first occurrence, preserve aliases; no outcome-dependent deduplication."""
    kept=[];aliases={}
    for record in records:
        action=np.asarray(record['action'],np.float32)
        assert action.shape==(2,) and np.isfinite(action).all() and np.max(np.abs(action))<=1
        match=next((x for x in kept if np.linalg.norm(action-np.asarray(x['action']))<=epsilon),None)
        if match is None:
            match=copy.deepcopy(record);match['candidate_id']=record['role'];match['roles']=[record['role']];kept.append(match)
        else:match['roles'].append(record['role'])
        aliases[record['role']]=match['candidate_id']
    return kept,aliases

def discounted_return(rewards,gamma=.995):
    result=0.
    for reward in reversed(rewards):result=float(reward)+gamma*result
    return result

def load_offline_evaluator():
    from types import SimpleNamespace
    from mpr_mpc._bootstrap import bootstrap
    bootstrap()
    from common.world_model import WorldModel
    from tdmpc2 import TDMPC2
    from planning.evaluator import TrajectoryWorldModelEvaluator
    from omegaconf import OmegaConf
    import evaluate_model_suite as suite
    p=load(HIST/'protocol.json');ck=Path(p['checkpoint'])
    raw=OmegaConf.load(p['source_environment_config']);raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False
    _,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[1]);cfg.obs_shape={'state':(259,)};cfg.action_dim=2;cfg.episode_length=1000
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');device=torch.device('cuda')
    model=WorldModel(cfg).to(device).eval().requires_grad_(False)
    agent=SimpleNamespace(model=model,cfg=cfg,device=device,discount=.995)
    TDMPC2.load(agent,{'model':torch.load(ck,map_location='cpu',weights_only=False)['model']})
    return TrajectoryWorldModelEvaluator(agent,10)
