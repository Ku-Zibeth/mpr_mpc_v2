"""Bounded origins, complete real returns: diagnostic, never a ranking approval."""
import argparse, contextlib, io, json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite

p=argparse.ArgumentParser();p.add_argument('--scenario',type=int,required=True);p.add_argument('--before',type=int,required=True)
p.add_argument('--variant',choices=['baseline','candidate'],required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
run=ROOT/'logs/benchmark/mpr_mpc_v311_sac_vs_sac_mppi_fixed100_20261109_202823'
raw=OmegaConf.load(run/'config_D_lattice_sac_mppi_seed1.yaml');raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_video=raw.save_agent=False;raw.residual_rl.buffer_size=8
ck=Path(json.load(open(run/'manifest.json'))['checkpoint']);_,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[a.scenario]);cfg.residual_rl['buffer_size']=8
set_seed(1);torch.set_float32_matmul_precision('high');env=make_env(cfg)
try:
 with contextlib.redirect_stdout(io.StringIO()):
  agent=suite._make_agent(cfg,env);loaded=agent.load(ck)
 agent.eval();suite._apply_evaluation_stack(agent.planner,'full_mpr_mpc',loaded)
 agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
 episode_seed=1+a.scenario;set_seed(episode_seed);agent.planner.reset(seed=episode_seed);obs=env.reset(seed=a.scenario)
 prefix=[]
 for line in next((run/'steps').glob('*_full_mpr_mpc_seed1.jsonl')).open():
  r=json.loads(line)
  if r['scenario']==a.scenario and r['episode_step']<a.before:prefix.append(r)
  if len(prefix)==a.before-1:break
 replay_error=0
 for r in prefix:
  replay_error=max(replay_error,abs(np.linalg.norm(env.unwrapped_metadrive.agent.velocity)-r['mpr/actual_speed_mps']))
  obs,_,done,info=env.step(np.asarray(r['executed_action'],dtype=np.float32))
  if done:raise RuntimeError('prefix terminated')
 if replay_error>1e-4:raise RuntimeError('source replay mismatch')
 initial_obs=obs.detach().cpu().numpy().copy() if torch.is_tensor(obs) else np.asarray(obs).copy()
 planner=agent.planner.local_mppi;captured=[];original=planner._candidate_batch
 def capture(*args):
  actions=original(*args);captured.append(actions.clone());return actions
 planner._candidate_batch=capture
 agent.act(obs,t0=False,eval_mode=True)
 result=agent.last_plan;baseline=result.refined_actions.clone();ctx=agent.planner.last_context.mppi_safety_context
 ev=planner.evaluator;z=ev.encode(obs)
 # Offline counterfactual pool only. Production skips performance proposals
 # while ranking is unqualified; sample the same SAC-centred Gaussian here
 # without declaring Value validated or changing any execution gate.
 if not captured:
  for _ in range(planner.iterations):
   planner._candidate_batch(baseline,planner.initial_std_vector.expand_as(baseline),baseline,planner.generator)
 pool=torch.cat(captured);pool=torch.unique(pool.flatten(1),dim=0).reshape(-1,*baseline.shape)
 predicted=ev.evaluate_mppi_candidates(z,pool,compute_cost=True)
 dev,smooth=planner._penalties(pool,baseline)
 # Current Cost is unvalidated and excluded, exactly as in production.
 preview=planner.candidate_safety.preview(pool.cpu(),ctx)
 score=predicted.total_value-planner.deviation_coef*dev-planner.smoothness_coef*smooth-planner.long_risk_coef*preview.risk_score.to(pool.device)
 order=score.argsort(descending=True).cpu().tolist();candidate=None
 for i in order[:2*planner.num_elites]:
  check=planner.candidate_safety.check(pool[i:i+1].cpu(),ctx)
  if bool(check.valid[0]) and bool(check.full_valid[0]) and not torch.equal(pool[i],baseline):
   candidate=pool[i].clone();break
 if candidate is None:raise RuntimeError('No full-checked local candidate for independent open-loop H-step audit')
 values=ev.evaluate_mppi_candidates(z,torch.stack((baseline,candidate)),compute_cost=False).total_value.cpu().tolist()
 sequence=baseline if a.variant=='baseline' else candidate
 obstacle_state=sorted([list(map(float,(*o.position,*o.velocity,o.heading,o.length,o.width))) for o in ctx.obstacles])
 origin_state=[*map(float,ctx.position),float(ctx.heading),float(ctx.speed),obstacle_state]
 start_route=float(getattr(ctx,'route_progress',0.) or 0.)
 # Full safe H-step open loop only for this counterfactual experiment. A
 # same-prefix recoverable-only sequence is NEVER executed open loop here.
 suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',loaded)
 gamma=float(agent.tdmpc_agent.discount);total=0.;discount=1.;actions=[];costs=[];rewards=[];events={'crash':False,'offroad':False};done=False;step=0
 while not done and step<1000:
  action=sequence[step] if step<planner.horizon else agent.act(obs,t0=False,eval_mode=True)
  obs,reward,done,info=env.step(action);step+=1
  total+=discount*float(reward);discount*=gamma
  actions.append(action.detach().cpu().tolist() if torch.is_tensor(action) else np.asarray(action).tolist())
  costs.append(float(info.get('cost',0.)));rewards.append(float(reward))
  events['crash'] |= any(bool(info.get(k,False)) for k in ('crash','crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk'))
  events['offroad'] |= bool(info.get('out_of_road',False))
 data=dict(scenario=a.scenario,before=a.before,variant=a.variant,checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
  observation=initial_obs.tolist(),origin_state=origin_state,origin_state_id=hashlib.sha256(json.dumps(origin_state).encode()).hexdigest(),
  prefix_sha256=hashlib.sha256(json.dumps([r['executed_action'] for r in prefix]).encode()).hexdigest(),replay_max_speed_error=replay_error,
  planned_actions=sequence.cpu().tolist(),predicted_values=values,predicted_gain=values[1]-values[0],
  actual_return=total,gamma=gamma,episode_complete=bool(done),horizon=planner.horizon,actual_controls=step,
  actions=actions,costs=costs,rewards=rewards,success=bool(info.get('success',False)),route_completion=float(info.get('route_completion',0.)),events=events,
  ranking_validated=False,note='Two origins are diagnostic only; no gate approval or paired success-rate claim.')
 a.output.write_text(json.dumps(data,indent=2));print(a.output,flush=True)
finally:env.close()
