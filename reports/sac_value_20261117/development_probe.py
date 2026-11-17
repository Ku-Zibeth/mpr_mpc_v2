"""Offline new-V search and four matched actual continuations; no default policy change."""
import argparse,contextlib,io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'))
from probe import clean,dump,snapshot
import numpy as np,torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite
from planning.local_mppi import LocalMPPI
from planning.sac_value import load_sac_value,policy_context_sha256

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scenario',type=int,required=True);p.add_argument('--before',type=int,required=True);p.add_argument('--choice',choices=['baseline','mean','best_elite','final_elite'],required=True);a=p.parse_args()
 meta=json.loads((R/'protocol.json').read_text());assert [a.scenario,a.before] in meta['targeted_dev_origins'];ck=Path(meta['checkpoint']);out=R/'development_counterfactual'/f'{a.scenario}_{a.before}';out.mkdir(parents=True,exist_ok=True);assert not (out/(a.choice+'.json')).exists()
 raw=OmegaConf.load(meta['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi;raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False
 raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8;raw.metadrive.simulator.horizon=1000
 _,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[a.scenario]);torch.set_float32_matmul_precision('high');set_seed(1);env=make_env(cfg)
 try:
  with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);loaded=agent.load(ck)
  agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',loaded);agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
  head,bundle=load_sac_value(R/'models/sac_value_candidate.pt',wm_checkpoint=ck,horizon=10,gamma=float(agent.tdmpc_agent.discount),device=agent.device,require_validated=False,policy_context=policy_context_sha256(cfg))
  agent.planner.evaluator.attach_offline_sac_value(head)
  seed=1+a.scenario;set_seed(seed);agent.planner.reset(seed=seed);obs=env.reset(seed=a.scenario)
  for i in range(a.before-1):
   act=agent.act(obs,t0=(i==0),eval_mode=True);obs,_,done,info=env.step(act);assert not done,'Prespecified SAC origin unreachable'
  agent.act(obs,t0=False,eval_mode=True);context=agent.planner.last_context;baseline=agent.last_plan.refined_actions.clone()
  options=dict(cfg.mpr_mpc['mppi']);options.update(terminal_value='sac_value',terminal_q_weight=1.,q_aggregation='ensemble_mean',temperature_definition='inverse',temperature=2.,audit_candidates=True,record_cost=True,cost_weight=0.)
  planner=LocalMPPI(agent.planner.evaluator,options,horizon=10,action_dim=2,device=agent.device);planner.reset(seed);torch.cuda.synchronize();started=time.perf_counter();plan=planner.plan(context.latent,baseline,eval_mode=True);torch.cuda.synchronize();planning_time_ms=1000*(time.perf_counter()-started)
  audit=plan.candidate_audit;names=['baseline','mean','best_elite','final_elite'];ids=[audit['indices'][n] for n in names];candidates=audit['actions'][ids]
  origin=snapshot(env,agent,obs,candidates,a.scenario,a.before);old=json.loads((Path(meta['old_data'])/f'dev_{a.scenario}_{a.before}'/'B_origin.json').read_text())
  for key in ('observation','state','rng','episode_config'):
   assert origin[key]==old[key],('Mismatch with old unmodified baseline',key)
  assert np.array_equal(baseline.cpu().numpy(),np.asarray(old['candidate_actions'][0],np.float32))
  dump(out/(a.choice+'_origin.json'),origin)
  if a.choice=='baseline':
   np.savez_compressed(out/'audit_samples.npz',actions=audit['actions'].cpu().numpy(),values=audit['values'].cpu().numpy(),
    iteration_actions=torch.stack(audit['iteration_actions']).cpu().numpy(),iteration_values=torch.stack(audit['iteration_values']).cpu().numpy(),
    original_elites=torch.stack(audit['original_elites']).cpu().numpy(),terminal_latents=audit['terminal_latents'].cpu().numpy(),costs=audit['costs'].cpu().numpy())
   dump(out/'audit.json',dict(indices=audit['indices'],choices=names,choice_scores=audit['values'][ids].cpu().tolist(),metrics=plan.metrics,planning_time_ms=planning_time_ms,definition=plan.evaluation,
    default_output='baseline' if plan.baseline_selected else 'mean',unique_prefixes=len({x[:10].cpu().numpy().tobytes() for x in candidates}),
    budget='64/8/4;262 complete WM trajectory evaluations;258 terminal-only cached re-scores, no additional reward/dynamics rollout'))
  controls=candidates[names.index(a.choice)];trace=[];done=False;crash=offroad=False;H=10
  while not done:
   i=len(trace);action=controls[i] if i<H else agent.act(obs,t0=False,eval_mode=True)
   obs,_,done,info=env.step(action);v=env.unwrapped_metadrive.agent
   trace.append(dict(action=action.cpu().tolist(),reward=float(info['tdmpc2_reward']),cost=float(info['cost']),position=list(map(float,v.position)),heading=float(v.heading_theta)))
   crash|=any(bool(info.get(k,False)) for k in ('crash','crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk'));offroad|=bool(info.get('out_of_road',False));assert len(trace)<=1000
  gamma=float(agent.tdmpc_agent.discount);ret=sum(gamma**i*x['reward'] for i,x in enumerate(trace));rtg=sum(gamma**i*x['reward'] for i,x in enumerate(trace[H:]));prefix=sum(gamma**i*x['reward'] for i,x in enumerate(trace[:H]));assert abs(ret-prefix-gamma**H*rtg)<1e-7
  dump(out/(a.choice+'.json'),dict(scenario=a.scenario,before=a.before,choice=a.choice,actions=controls.cpu().tolist(),trace=trace,
   actual_return=ret,episode_return=sum(x['reward'] for x in trace),actual_continuation=rtg,actual_prefix=prefix,
   success=bool(info['success']),crash=crash,out_of_road=offroad,route_completion=float(info['route_completion']),cost=sum(x['cost'] for x in trace),
   length=len(trace),prefix_complete=len(trace)>=H,predicted_return=float(audit['values'][ids[names.index(a.choice)]]),
   predicted_cost=float(audit['costs'][ids[names.index(a.choice)]]),planning_time_ms=planning_time_ms,gamma=gamma,
   score_value_checkpoint='sac_value_candidate.pt',value_used_only_in_offline_prefix_search=True,default_online_output_unchanged=True))
  print(a.scenario,a.before,a.choice,ret,flush=True)
 finally:env.close()
if __name__=='__main__':main()
