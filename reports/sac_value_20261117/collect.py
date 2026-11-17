"""Real frozen-policy data; primary labels are SAC RTG after executed H prefixes."""
import argparse,contextlib,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'))
from probe import snapshot,clean,dump
import numpy as np,torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite
from planning.local_mppi import LocalMPPI
from planning.sac_value import policy_context_sha256

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scenario',type=int,required=True);p.add_argument('--before',type=int,required=True);p.add_argument('--branch',choices=['B','gaussian_1','gaussian_2'],required=True);a=p.parse_args()
 meta=json.loads((R/'protocol.json').read_text());assert a.scenario in meta['training_scenarios'];ck=Path(meta['checkpoint']);out=R/'training_data'/f'{a.scenario}_{a.before}';out.mkdir(parents=True,exist_ok=True);file=out/(a.branch+'.npz');assert not file.exists()
 raw=OmegaConf.load(meta['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi;raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False
 raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8;raw.metadrive.simulator.horizon=1000
 _,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[a.scenario]);torch.set_float32_matmul_precision('high');set_seed(cfg.seed);env=make_env(cfg)
 try:
  with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);state=agent.load(ck)
  agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',state);agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
  seed=1+a.scenario;set_seed(seed);agent.planner.reset(seed=seed);obs=env.reset(seed=a.scenario);assert int(env.unwrapped_metadrive.current_seed)==a.scenario
  for i in range(a.before-1):
   action=agent.act(obs,t0=(i==0),eval_mode=True);obs,_,done,info=env.step(action)
   if done:dump(out/('unavailable_'+a.branch+'.json'),dict(reason='SAC terminated before origin',scenario=a.scenario,before=a.before,step=i+1));return
  agent.act(obs,t0=False,eval_mode=True);baseline=agent.last_plan.refined_actions.clone();z0=agent.planner.last_context.latent.clone();H=cfg.horizon
  sampler=LocalMPPI(agent.planner.evaluator,dict(cfg.mpr_mpc['mppi']),horizon=H,action_dim=2,device=agent.device);sampler.reset(seed)
  pool=sampler._candidate_batch(baseline,sampler.initial_std_vector.expand_as(baseline),baseline)[:3].clone();pool[:,-1]=baseline[-1]
  origin=snapshot(env,agent,obs,pool,a.scenario,a.before);dump(out/(a.branch+'_origin.json'),origin)
  index=meta['train_branches'].index(a.branch);prefix=pool[index];observations=[obs.cpu().numpy().copy()];actions=[];rewards=[];costs=[];terminated=[];truncated=[];sac_state=None;done=False;crash=offroad=False
  while not done:
   i=len(actions);action=prefix[i] if i<H else agent.act(obs,t0=False,eval_mode=True)
   if i==H:sac_state=agent.planner.last_context.residual_state.cpu().numpy().reshape(-1)
   obs,reward,done,info=env.step(action);observations.append(obs.cpu().numpy().copy());actions.append(action.cpu().numpy().copy());rewards.append(float(info['tdmpc2_reward']));costs.append(float(info['cost']))
   terminated.append(bool(info.get('terminated',False)));truncated.append(bool(info.get('truncated',False)))
   crash|=any(bool(info.get(k,False)) for k in ('crash','crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk'));offroad|=bool(info.get('out_of_road',False))
   assert len(actions)<=1000
  gamma=float(agent.tdmpc_agent.discount);rtg=np.zeros(len(rewards)+1,np.float64)
  for i in range(len(rewards)-1,-1,-1):rtg[i]=rewards[i]+gamma*rtg[i+1]
  observations=np.asarray(observations,np.float32);actions=np.asarray(actions,np.float32)
  latents=torch.cat([agent.planner.evaluator.encode(torch.tensor(observations[i:i+256])) for i in range(0,len(observations),256)],0)
  starts=list(range(0,max(0,len(actions)-H+1),meta['window_stride']))
  if len(actions)>=H and len(actions)-H not in starts:starts.append(len(actions)-H)
  imagined=[]
  for j in range(0,len(starts),256):
   indices=np.asarray(starts[j:j+256]);z=latents[indices].clone()
   for t in range(H):z=agent.tdmpc_agent.model.next(z,torch.tensor(actions[indices+t],device=agent.device),None)
   imagined.append(z.cpu().numpy())
  starts=np.asarray(starts,np.int64);predicted=np.concatenate(imagined) if imagined else np.empty((0,512),np.float32)
  np.savez_compressed(file,observations=observations,actions=actions,rewards=np.asarray(rewards),costs=np.asarray(costs),terminated=np.asarray(terminated),truncated=np.asarray(truncated),
   starts=starts,predicted_latents=predicted,actual_latents=latents[starts+H].cpu().numpy(),targets=rtg[starts+H],rtg=rtg,
   origin_predicted=predicted[0] if len(starts) else np.empty(0),origin_actual=latents[H].cpu().numpy() if len(actions)>=H else np.empty(0),
   terminal_sac_state=sac_state if sac_state is not None else np.empty(0),gamma=np.array(gamma))
  dump(out/(a.branch+'_meta.json'),dict(scenario=a.scenario,before=a.before,branch=a.branch,origin_id=out.name,samples=len(starts),
   length=len(actions),native_done=True,prefix_complete=len(actions)>=H,success=bool(info['success']),crash=crash,out_of_road=offroad,
   route_completion=float(info['route_completion']),return_sum=sum(rewards),discounted_return=float(rtg[0]),cost_sum=sum(costs),gamma=gamma,
   policy_context_sha256=policy_context_sha256(cfg),wm_checkpoint_sha256=meta['checkpoint_sha256']))
  print(out.name,a.branch,'steps',len(actions),'samples',len(starts),flush=True)
 finally:env.close()
if __name__=='__main__':main()
