"""Offline candidate-return probe, not an optimizer gate or safety controller."""
import argparse,json,hashlib,sys,contextlib,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite
p=argparse.ArgumentParser();p.add_argument('--scenario',type=int,required=True);p.add_argument('--before',type=int,required=True)
p.add_argument('--variant',choices=['sac','mean'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
meta=json.load(open(Path(__file__).parent/'paired_run.json'));ck=Path(meta['checkpoint'])
raw=OmegaConf.load(meta['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_video=raw.save_agent=False;raw.residual_rl.buffer_size=8
_,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[a.scenario]);cfg.residual_rl['buffer_size']=8
torch.set_float32_matmul_precision('high');set_seed(cfg.seed);env=make_env(cfg)
try:
 with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);loaded=agent.load(ck)
 agent.eval();suite._apply_evaluation_stack(agent.planner,'full_mpr_mpc',loaded)
 agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step));seed=int(cfg.seed)+a.scenario
 set_seed(seed);agent.planner.reset(seed=seed);obs=env.reset(seed=a.scenario)
 assert int(env.unwrapped_metadrive.current_seed)==a.scenario
 archived=ROOT/'logs/benchmark/mpr_mpc_v311_sac_vs_sac_mppi_fixed100_20261109_202823/steps'
 prefix=[]
 for line in next(archived.glob('*_full_mpr_mpc_seed1.jsonl')).open():
  r=json.loads(line)
  if r['scenario']==a.scenario and r['episode_step']<a.before:prefix.append(r)
  if len(prefix)==a.before-1:break
 assert len(prefix)==a.before-1
 replay_error=0.
 for r in prefix:
  replay_error=max(replay_error,abs(np.linalg.norm(env.unwrapped_metadrive.agent.velocity)-r['mpr/actual_speed_mps']))
  obs,_,done,info=env.step(np.asarray(r['executed_action'],np.float32))
  if done:raise RuntimeError('Prefix terminated before origin')
 if replay_error>1e-4:raise RuntimeError(f'Prefix mismatch: {replay_error}')
 initial_obs=obs.cpu().tolist();agent.act(obs,t0=False,eval_mode=True);r=agent.last_plan;c=agent.planner.last_context
 baseline=r.refined_actions;mean=r.debug['final_mppi_mean'].to(agent.device)
 prediction=agent.planner.evaluator.evaluate_mppi_candidates(c.latent,torch.stack((baseline,mean)),compute_cost=False)
 v=prediction.total_value.cpu().tolist()
 from common import math as wm_math
 members=wm_math.two_hot_inv(agent.tdmpc_agent.model.Q(prediction.terminal_latent,torch.stack((baseline,mean))[:,-1],None,return_type='all'),cfg).squeeze(-1).cpu().tolist()
 decomposition=dict(discounted_reward=prediction.discounted_reward.cpu().tolist(),
  terminal_q=prediction.terminal_q.cpu().tolist(),discounted_terminal_contribution=(prediction.total_value-prediction.discounted_reward).cpu().tolist(),
  terminal_Q_members=members,terminal_Q_member_deltas=[row[1]-row[0] for row in members])
 vehicle=env.unwrapped_metadrive.agent
 obstacles=agent.planner.structured.planner._collect_obstacles(vehicle)
 state=dict(ego=[*map(float,vehicle.position),*map(float,vehicle.velocity),float(vehicle.heading_theta)],
  obstacles=sorted([list(map(float,(*o.position,*o.velocity,o.heading,o.length,o.width))) for o in obstacles]))
 actions=baseline if a.variant=='sac' else mean
 suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',loaded)
 gamma=float(agent.tdmpc_agent.discount);discount=1.;ret=0.;trace=[];done=False;i=0;crash=offroad=False
 wm_z=c.latent.detach().clone();latent_errors=[];true_state_Q=None
 while not done and i<1000:
  # H+1 physical commands: a_H is the first action contained in Q(z_H,a_H).
  # In the WM evaluator itself there are still exactly H next()/reward() calls.
  action=actions[i] if i<len(actions) else agent.act(obs,t0=False,eval_mode=True)
  if i<cfg.horizon:
   with torch.no_grad():wm_z=agent.tdmpc_agent.model.next(wm_z,action[None].to(agent.device),None)
  obs,reward,done,info=env.step(action);i+=1;ret+=discount*float(reward);discount*=gamma
  if i<=cfg.horizon:
   with torch.no_grad():actual_z=agent.planner.evaluator.encode(obs)
   latent_errors.append(float((wm_z-actual_z).square().mean().sqrt()))
   if i==cfg.horizon:
    with torch.no_grad():true_state_Q=wm_math.two_hot_inv(agent.tdmpc_agent.model.Q(actual_z,actions[cfg.horizon:cfg.horizon+1],None,return_type='all'),cfg).squeeze().cpu().tolist()
  crash |= any(bool(info.get(k,False)) for k in ('crash','crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk'))
  offroad |= bool(info.get('out_of_road',False))
  trace.append(dict(action=action.cpu().tolist(),reward=float(reward),cost=float(info.get('cost',0.))))
 result=dict(scenario=a.scenario,before=a.before,variant=a.variant,observation=initial_obs,observed_state=state,
  observed_state_id=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest(),replay_max_speed_error=replay_error,
  predicted_values=v,predicted_gain=v[1]-v[0],decomposition=decomposition,latent_rollout_rmse=latent_errors,true_state_terminal_Q_members=true_state_Q,
  actual_first_H_discounted_reward=sum(gamma**j*r['reward'] for j,r in enumerate(trace[:cfg.horizon])),
  actual_terminal_continuation=sum(gamma**j*r['reward'] for j,r in enumerate(trace[cfg.horizon:])),baseline=baseline.cpu().tolist(),mean=mean.cpu().tolist(),
  actual_discounted_return=ret,gamma=gamma,episode_complete=bool(done),controls=i,success=bool(info.get('success',False)),
  route_completion=float(info.get('route_completion',0.)),crash=crash,offroad=offroad,trace=trace,
  checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
  scope='H+1 actual candidate commands then the same SAC continuation to termination. Recorded observation/ego/collected-obstacle state match is checked across fresh processes; not a claim of full hidden simulator-state equivalence or a broad ranking validation. No runtime gate is installed.')
 assert result['episode_complete'];a.output.write_text(json.dumps(result,indent=2));print(a.output,flush=True)
finally:env.close()
