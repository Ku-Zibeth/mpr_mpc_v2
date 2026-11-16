"""Matched-prefix counterfactuals; all diagnostics stay outside online selection."""
import argparse,contextlib,hashlib,io,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite
from planning.local_mppi import LocalMPPI
from planning.mppi_ablation import terminal_weight_ablation_config
from tools.value_calibration import PrefixValueProbe,terminal_q_audit,digest,assert_origin_match

def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 return x

def dump(path,x):path.write_text(json.dumps(clean(x),indent=2,allow_nan=False))
def rng_state(obj):
 if obj is None:return None
 rng=getattr(obj,'np_random',None)
 if rng is None:return None
 if hasattr(rng,'get_state'):return clean(rng.get_state())
 if hasattr(rng,'bit_generator'):return clean(rng.bit_generator.state)
 return type(rng).__name__

def snapshot(env,agent,obs,actions,scenario,before):
 sim=env.unwrapped_metadrive;v=sim.agent;traffic=[]
 for obj in sim.engine.get_objects().values():
  if not hasattr(obj,'velocity') or not hasattr(obj,'position'):continue
  traffic.append([type(obj).__name__,*map(float,np.asarray(obj.position)[:2]),*map(float,np.asarray(obj.velocity)[:2]),float(getattr(obj,'heading_theta',0.))])
 state=dict(ego=[*map(float,v.position),*map(float,v.velocity),float(v.heading_theta)],traffic=sorted(traffic))
 rng=dict(python=digest(clean(random.getstate())),numpy=digest(clean(np.random.get_state())),
  torch=digest(torch.random.get_rng_state().tolist()),cuda=digest(torch.cuda.get_rng_state().tolist()),
  engine=rng_state(sim.engine),traffic_manager=rng_state(getattr(sim.engine,'traffic_manager',None)))
 lane=getattr(v,'lane',None);coverage=dict(speed=float(np.linalg.norm(v.velocity)),lane_type=type(lane).__name__,traffic_count=max(0,len(traffic)-1))
 if lane is not None:
  s,d=lane.local_coordinates(v.position);coverage.update(lateral=float(d),lane_heading_change_10m=float(lane.heading_theta_at(min(lane.length,s+10))-lane.heading_theta_at(s)))
 return dict(episode_config=dict(scenario=scenario,before=before,horizon=1000,seed=1+scenario,simulator=agent.cfg.metadrive['simulator']),
  observation=obs.cpu().tolist(),state=state,rng=rng,candidate_actions=actions.cpu().tolist(),coverage=coverage)

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scenario',type=int,required=True);p.add_argument('--before',type=int,required=True);p.add_argument('--legacy',action='store_true');p.add_argument('--branch',required=True);a=p.parse_args()
 meta=json.loads((R/'protocol.json').read_text());ck=Path(meta['checkpoint']);split='diagnosis' if a.legacy else 'dev' if a.scenario in meta['dev_scenarios'] else 'validation'
 out=R/'probes'/f'{split}_{a.scenario}_{a.before}';out.mkdir(parents=True,exist_ok=True);assert not (out/(a.branch+'.json')).exists()
 raw=OmegaConf.load(meta['source_environment_config']);base=OmegaConf.to_container(OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi,resolve=True)
 raw.mpr_mpc.mppi=OmegaConf.create(terminal_weight_ablation_config(base,'Q1'));raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False
 raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8;raw.metadrive.simulator.horizon=1000
 _,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[a.scenario]);set_seed(cfg.seed);torch.set_float32_matmul_precision('high');env=make_env(cfg)
 try:
  with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);loaded=agent.load(ck)
  agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',loaded);agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
  if a.legacy:
   old=json.loads((ROOT/f'reports/mppi_value_20261116/probe_{a.scenario}_sac.json').read_text());prefix=old['prefix_controls']
  reference=None;candidate_actions=None;predictions=None;results={};search={};labels=[];physics=None
  branches=['B','legacy_mean'] if a.legacy else ['B',*meta['groups']]+(['elite_Q1'] if a.scenario in meta['elite_scenarios'] else [])
  assert a.branch in branches
  for branch in [a.branch]:
   seed=1+a.scenario;set_seed(seed);agent.planner.reset(seed=seed);obs=env.reset(seed=a.scenario);assert int(env.unwrapped_metadrive.current_seed)==a.scenario
   replay_error=0.;prefix_actions=[]
   for i in range(a.before-1):
    action=torch.tensor(prefix[i],device=agent.device) if a.legacy else agent.act(obs,t0=(i==0),eval_mode=True)
    prefix_actions.append(action.cpu().tolist());obs,_,done,info=env.step(action)
    if done:
     dump(out/('unavailable_'+branch+'.json'),dict(scenario=a.scenario,before=a.before,reason='SAC prefix terminated before prescribed origin',prefix_steps=i+1,info=info));print('unreachable',out,flush=True);return
   agent.act(obs,t0=False,eval_mode=True);context=agent.planner.last_context;baseline=agent.last_plan.refined_actions.clone();z=context.latent.clone()
   if candidate_actions is None:
    pool=[baseline];labels=['B']
    if a.legacy:
     assert np.array_equal(obs.cpu().numpy(),np.asarray(old['observation'],dtype=np.float32))
     assert np.array_equal(baseline.cpu().numpy(),np.asarray(old['baseline'],dtype=np.float32))
     pool.append(torch.tensor(old['mean'],device=agent.device));labels.append('legacy_mean')
    else:
     for name in meta['groups']:
      planner=LocalMPPI(agent.planner.evaluator,terminal_weight_ablation_config(base,name),horizon=cfg.horizon,action_dim=cfg.action_dim,device=agent.device);planner.reset(seed)
      with torch.random.fork_rng(devices=[torch.cuda.current_device()]):plan=planner.plan(z,baseline,eval_mode=True)
      pool.append(plan.final_mean);labels.append(name);search[name]=dict(metrics=plan.metrics,iterations=plan.iterations,definition=plan.evaluation)
     if 'elite_Q1' in branches:
      elites=torch.tensor([it['best_elite_actions'] for it in search['Q1']['iterations']],device=agent.device)
      scores=agent.planner.evaluator.evaluate_mppi_candidates(z,elites,q_aggregation='ensemble_mean',compute_cost=False).total_value
      pool.append(elites[scores.argmax()]);labels.append('elite_Q1')
    candidate_actions=torch.stack(pool)
    predictions=agent.planner.evaluator.evaluate_mppi_candidates(z,candidate_actions,q_aggregation='ensemble_mean',compute_cost=False,capture_ensemble=True)
    if agent.planner.safety is not None and context.safety_context is not None:
     xy,yaw,speed=agent.planner.safety.rollout(candidate_actions,context.safety_context);stride=agent.planner.safety.substeps
     physics=dict(xy=xy[:,stride::stride].cpu().numpy(),yaw=yaw[:,stride::stride].cpu().numpy(),speed=speed[:,stride::stride].cpu().numpy(),
      approximation='Existing SAC kinematic feature model only. Config acceleration/braking bounds are not calibrated truth. Never used in online candidate selection.')
   else:
    assert torch.equal(baseline,candidate_actions[0])
   origin=snapshot(env,agent,obs,candidate_actions,a.scenario,a.before)
   if reference is None:reference=origin;dump(out/(branch+'_origin.json'),dict(origin,prefix_actions=prefix_actions))
   else:assert_origin_match(reference,origin)
   idx=labels.index(branch);actions=candidate_actions[idx];probe=PrefixValueProbe(agent.planner.evaluator,z);trace=[];terminal=None;crash=offroad=False;done=False;H=cfg.horizon
   while not done and len(trace)<1000:
    i=len(trace);action=actions[i] if i<H else agent.act(obs,t0=False,eval_mode=True)
    if i==H:
     terminal=terminal_q_audit(agent.tdmpc_agent,probe.imagined,probe.actual,actions[H:H+1],action[None])
    if i<H:probe.before(obs,action)
    obs,reward,done,info=env.step(action);actual=float(info['tdmpc2_reward'])
    if i<H:
     row=probe.after(obs,info)
     if physics is not None:
      v=env.unwrapped_metadrive.agent;angle=physics['yaw'][idx,i]-float(v.heading_theta)
      row.update(position_error_m=float(np.linalg.norm(physics['xy'][idx,i]-v.position)),speed_error_mps=float(physics['speed'][idx,i]-np.linalg.norm(v.velocity)),heading_error_rad=float(np.arctan2(np.sin(angle),np.cos(angle))))
    crash|=any(bool(info.get(k,False)) for k in ('crash','crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk'));offroad|=bool(info.get('out_of_road',False))
    trace.append(dict(action=action.cpu().tolist(),reward=actual,returned_float32_reward=float(reward),metadrive_reward=float(info['metadrive_reward']),cost=float(info.get('cost',0.))))
   assert done;gamma=float(agent.tdmpc_agent.discount);prefix_return=sum(gamma**i*r['reward'] for i,r in enumerate(trace[:H]));rtg=sum(gamma**i*r['reward'] for i,r in enumerate(trace[H:]));ret=sum(gamma**i*r['reward'] for i,r in enumerate(trace));assert abs(ret-prefix_return-gamma**H*rtg)<1e-7
   result=dict(branch=branch,matched_observed_origin=True,origin_digest=digest(clean(reference)),actions=actions.cpu().tolist(),
    prefix=probe.records,terminal=terminal,actual_prefix=prefix_return,actual_continuation=rtg,actual_discounted_return=ret,episode_return=sum(r['reward'] for r in trace),trace=trace,
    success=bool(info.get('success',False)),crash=crash,out_of_road=offroad,route_completion=float(info.get('route_completion',0.)),length=len(trace),gamma=gamma,
    predicted_prefix=float(predictions.discounted_reward[idx]),terminal_discount=float(predictions.terminal_discount[idx]),predicted_q=float(predictions.terminal_q[idx]),
    predicted_q_members=predictions.q_member_values[:,idx].cpu().tolist(),predicted_terminal_latent=predictions.terminal_latent[idx].cpu().tolist())
   results[branch]=result;dump(out/(branch+'.json'),result);print(a.scenario,a.before,branch,len(trace),ret,flush=True)
  dump(out/(a.branch+'_summary.json'),dict(scenario=a.scenario,before=a.before,split=split,labels=labels,search=search,origin_digest=digest(clean(reference)),coverage=reference['coverage'],
    physics_source=physics['approximation'] if physics else None,checkpoint_sha256=meta['checkpoint_sha256'],
    scope='Observed/traffic/RNG/config/candidates matched exactly across reset branches. Not a full Bullet/hidden-controller snapshot. Exactly H physical commands then unchanged SAC continuation. Initial 535-D SAC state unchanged. No origin is selected by achieved return.'))
 finally:env.close()
if __name__=='__main__':main()
