"""Four reward-blind Q-MPPI proposals, followed by native frozen SAC execution."""
import argparse,contextlib,io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'))
from probe import snapshot,dump
import numpy as np,torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite
from planning.local_mppi import LocalMPPI
from planning.sac_value import policy_context_sha256
from planning.candidate_value import aligned_returns

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scenario',type=int,required=True);ap.add_argument('--before',type=int,required=True);ap.add_argument('--branch',required=True);a=ap.parse_args()
 p=json.loads((R/'protocol.json').read_text());assert a.scenario in p['new_fit_scenarios']+p['new_stop_scenarios']+p['validation_scenarios'];assert a.branch in p['branches'];ck=Path(p['checkpoint']);out=R/'data'/f'{a.scenario}_{a.before}';out.mkdir(parents=True,exist_ok=True)
 if ((out/(a.branch+'.npz')).exists() and (out/(a.branch+'_meta.json')).exists()) or (out/('unavailable_'+a.branch+'.json')).exists():
  print('Already completed; retained',a.scenario,a.before,a.branch,flush=True);return
 raw=OmegaConf.load(p['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi;raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False
 raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8;raw.metadrive.simulator.horizon=1000
 _,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[a.scenario]);torch.set_num_threads(1);torch.set_float32_matmul_precision('high');set_seed(1);env=make_env(cfg)
 try:
  with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);state=agent.load(ck)
  agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',state);agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
  seed=1+a.scenario;set_seed(seed);agent.planner.reset(seed=seed);obs=env.reset(seed=a.scenario);assert int(env.unwrapped_metadrive.current_seed)==a.scenario
  for i in range(a.before-1):
   action=agent.act(obs,t0=(i==0),eval_mode=True);obs,_,done,info=env.step(action)
   if done:
    dump(out/('unavailable_'+a.branch+'.json'),dict(reason='native SAC terminated before prescribed origin; never replaced based on reward',scenario=a.scenario,before=a.before,steps=i+1,success=bool(info['success']),out_of_road=bool(info.get('out_of_road',False))));return
  agent.act(obs,t0=False,eval_mode=True);baseline=agent.last_plan.refined_actions.clone();z0=agent.planner.last_context.latent.clone();H=int(cfg.horizon);assert H==10
  options=dict(cfg.mpr_mpc['mppi']);options.update(audit_candidates=True,record_cost=False,cost_weight=0.)
  planner=LocalMPPI(agent.planner.evaluator,options,horizon=H,action_dim=2,device=agent.device);planner.reset(seed)
  torch.cuda.synchronize();begin=time.perf_counter();plan=planner.plan(z0,baseline,eval_mode=True);torch.cuda.synchronize();ms=1000*(time.perf_counter()-begin)
  audit=plan.candidate_audit;assert not plan.numerical_error
  ids=[0,2,1,audit['indices']['best_elite']];pool=audit['actions'][ids];idx=p['branches'].index(a.branch);commands=pool[idx]
  origin=snapshot(env,agent,obs,pool,a.scenario,a.before);dump(out/(a.branch+'_origin.json'),origin)
  if a.branch=='baseline':
   np.savez_compressed(out/'audit.npz',actions=audit['actions'].cpu().numpy(),scores=audit['values'].cpu().numpy(),latents=audit['terminal_latents'].cpu().numpy(),iteration_actions=torch.stack(audit['iteration_actions']).cpu().numpy(),iteration_scores=torch.stack(audit['iteration_values']).cpu().numpy())
   dump(out/'audit.json',dict(indices=ids,branches=p['branches'],metrics=plan.metrics,planning_time_ms=ms,definition=plan.evaluation,default_output='baseline' if plan.baseline_selected else 'mean',selected_action_consistent=bool(torch.equal(plan.action,plan.selected_sequence[0]))))
  observations=[obs.cpu().numpy().copy()];actions=[];rewards=[];costs=[];dones=[];terms=[];truncs=[];xy=[];speeds=[];done=crash=offroad=False
  while not done:
   i=len(actions);act=commands[i] if i<H else agent.act(obs,t0=False,eval_mode=True)
   obs,_,done,info=env.step(act);v=env.unwrapped_metadrive.agent
   observations.append(obs.cpu().numpy().copy());actions.append(act.cpu().numpy().copy());rewards.append(float(info['tdmpc2_reward']));costs.append(float(info['cost']));dones.append(bool(done));terms.append(bool(info.get('terminated',False)));truncs.append(bool(info.get('truncated',False)));xy.append(np.asarray(v.position).copy());speeds.append(float(np.linalg.norm(v.velocity)))
   crash|=any(bool(info.get(k,False)) for k in ('crash','crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk'));offroad|=bool(info.get('out_of_road',False));assert len(actions)<=1000
  observations=np.asarray(observations,np.float32);actions=np.asarray(actions,np.float32);gamma=float(agent.tdmpc_agent.discount);assert gamma==p['gamma'];labels=aligned_returns(rewards,[0],horizon=H,gamma=gamma,done=dones)
  complete=bool(labels['valid_terminal'][0]);actual_z=agent.planner.evaluator.encode(torch.tensor(observations[min(H,len(actions))])).cpu().numpy().reshape(-1)
  np.savez_compressed(out/(a.branch+'.npz'),observations=observations,actions=actions,rewards=rewards,costs=costs,done=dones,terminated=terms,truncated=truncs,positions=xy,speeds=speeds,rtg=labels['rtg'],
   z0=z0.cpu().numpy().reshape(-1),predicted_terminal=audit['terminal_latents'][ids[idx]].cpu().numpy(),actual_terminal=actual_z if complete else np.empty(0),planned_actions=commands.cpu().numpy(),baseline_actions=baseline.cpu().numpy(),
   predicted_return=np.array(float(audit['values'][ids[idx]])),predicted_prefix=np.array(float(audit['discounted_rewards'][ids[idx]])),terminal_discount=np.array(float(audit['terminal_discounts'][ids[idx]])),gamma=np.array(gamma))
  dump(out/(a.branch+'_meta.json'),dict(scenario=a.scenario,before=a.before,branch=a.branch,native_done=True,length=len(actions),prefix_complete=complete,success=bool(info['success']),crash=crash,out_of_road=offroad,route_completion=float(info['route_completion']),episode_return=sum(rewards),actual_return=float(labels['total'][0]),actual_continuation=float(labels['continuation'][0]) if complete else None,cost_sum=sum(costs),planning_time_ms=ms,average_speed=float(np.mean(speeds)),policy_context_sha256=policy_context_sha256(cfg),wm_checkpoint_sha256=p['checkpoint_sha256'],end_reason=info.get('end_reason','')))
  print(a.scenario,a.before,a.branch,len(actions),float(labels['total'][0]),flush=True)
 finally:env.close()
if __name__=='__main__':
 import fcntl
 scenario=sys.argv[sys.argv.index('--scenario')+1];before=sys.argv[sys.argv.index('--before')+1];branch=sys.argv[sys.argv.index('--branch')+1]
 locks=R/'locks';locks.mkdir(exist_ok=True)
 with (locks/f'{scenario}_{before}_{branch}.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX);main()
