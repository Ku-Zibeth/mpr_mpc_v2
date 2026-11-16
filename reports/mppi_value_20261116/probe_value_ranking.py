"""Offline common-origin/value audit; no online planner gate, new controller or training."""
import argparse, contextlib, hashlib, io, itertools, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np, torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from common import math as wm_math
from env import make_env
import evaluate_model_suite as suite
from planning.local_mppi import LocalMPPI
from planning.mppi_ablation import VALUE_ABLATIONS, value_ablation_config
REPORT=Path(__file__).parent

def ranking_audit(values):
 pairs=list(itertools.combinations(range(5),2));values=np.asarray(values)
 scores=np.array([values[list(pair)].mean(0) for pair in pairs]);n=values.shape[1]
 left,right=np.triu_indices(n,1);comparisons=[]
 def ranks(x):
  order=np.argsort(x);r=np.empty(len(x),float);start=0
  while start<len(x):
   stop=start+1
   while stop<len(x) and x[order[stop]]==x[order[start]]:stop+=1
   r[order[start:stop]]=(start+stop-1)/2;start=stop
  return r
 for i,j in itertools.combinations(range(len(pairs)),2):
  a,b=scores[i],scores[j];da,db=a[left]-a[right],b[left]-b[right]
  valid=(np.abs(da)>1e-6)&(np.abs(db)>1e-6)
  ra,rb=ranks(a),ranks(b);rho=float(np.corrcoef(ra,rb)[0,1]) if np.std(ra)>0 and np.std(rb)>0 else None
  comparisons.append(dict(pair_a=pairs[i],pair_b=pairs[j],joint_non_tied=int(valid.sum()),
   ranking_agreement=float(np.mean(np.sign(da[valid])==np.sign(db[valid]))) if valid.any() else None,
   spearman=rho,top8_overlap=len(set(np.argsort(a)[-8:])&set(np.argsort(b)[-8:]))/min(8,n)))
 return dict(q_pairs=pairs,candidate_count=n,comparisons=comparisons,
  mean_ranking_agreement=float(np.mean([r['ranking_agreement'] for r in comparisons if r['ranking_agreement'] is not None])),
  mean_spearman=float(np.mean([r['spearman'] for r in comparisons if r['spearman'] is not None])),
  mean_top8_overlap=float(np.mean([r['top8_overlap'] for r in comparisons])),
  ensemble_std_mean=float(values.std(0).mean()),ensemble_std_p95=float(np.percentile(values.std(0),95)))

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scenario',type=int,required=True);p.add_argument('--before',type=int,required=True)
 p.add_argument('--variant',choices=['sac','mean'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 meta=json.loads((REPORT/'protocol.json').read_text());ck=Path(meta['checkpoint'])
 raw=OmegaConf.load(meta['source_environment_config']);base=OmegaConf.to_container(OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi,resolve=True)
 raw.mpr_mpc.mppi=OmegaConf.create(value_ablation_config(base,'D0'))
 raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_video=raw.save_agent=False;raw.residual_rl.buffer_size=8
 _,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[a.scenario]);torch.set_float32_matmul_precision('high');set_seed(cfg.seed);env=make_env(cfg)
 try:
  with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);loaded=agent.load(ck)
  agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',loaded)
  agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step));seed=(int(cfg.seed)+a.scenario)%2**32
  set_seed(seed);agent.planner.reset(seed=seed);obs=env.reset(seed=a.scenario)
  assert int(env.unwrapped_metadrive.current_seed)==a.scenario
  archived_origin=(a.scenario,a.before) in {(1714,250),(559,600)};prefix=[];replay_error=0.
  if archived_origin:
   archived=ROOT/'logs/benchmark/mpr_mpc_v311_sac_vs_sac_mppi_fixed100_20261109_202823/steps'
   for line in next(archived.glob('*_full_mpr_mpc_seed1.jsonl')).open():
    r=json.loads(line)
    if r['scenario']==a.scenario and r['episode_step']<a.before:prefix.append(r)
    if len(prefix)==a.before-1:break
   assert len(prefix)==a.before-1
  for i in range(a.before-1):
   if archived_origin:
    r=prefix[i];replay_error=max(replay_error,abs(np.linalg.norm(env.unwrapped_metadrive.agent.velocity)-r['mpr/actual_speed_mps']))
    action=np.asarray(r['executed_action'],np.float32)
   else:
    action=agent.act(obs,t0=(i==0),eval_mode=True);prefix.append(dict(executed_action=action.cpu().tolist()))
   obs,_,done,info=env.step(action)
   if done:raise RuntimeError('Prefix terminated before origin')
  if replay_error>1e-4:raise RuntimeError(f'Prefix mismatch: {replay_error}')
  initial_obs=obs.cpu().tolist();agent.act(obs,t0=False,eval_mode=True);baseline=agent.last_plan.refined_actions.clone();z=agent.planner.last_context.latent.clone()
  evaluator=agent.planner.evaluator;device=agent.device;cuda=[torch.device(device).index or 0] if torch.device(device).type=='cuda' else []
  results={};pool=[baseline];labels=['sac'];indices={}
  def add(tensor,label):
   for i,t in enumerate(pool):
    if torch.equal(tensor,t):return i
   pool.append(tensor.clone());labels.append(label);return len(pool)-1
  # Same initial global RNG and Gaussian innovations for each one-factor search.
  for name in VALUE_ABLATIONS:
   search=LocalMPPI(evaluator,value_ablation_config(base,name),horizon=cfg.horizon,action_dim=cfg.action_dim,device=device);search.reset(seed)
   with torch.random.fork_rng(devices=cuda):r=search.plan(z,baseline,eval_mode=True)
   results[name]=dict(metrics=r.metrics,iterations=r.iterations,evaluation=r.evaluation)
   indices[name]=dict(mean=add(r.final_mean,name+'_mean'),elites=[add(torch.tensor(it['best_elite_actions'],device=device),name+f'_elite_{j}') for j,it in enumerate(r.iterations)])
  # A shared 64-proposal batch, not just the five already-selected means.
  search=LocalMPPI(evaluator,value_ablation_config(base,'D0'),horizon=cfg.horizon,action_dim=cfg.action_dim,device=device);search.reset(seed)
  std=search.initial_std_vector.expand_as(baseline)
  for j,t in enumerate(search._candidate_batch(baseline,std,baseline)[1:]):add(t,f'shared_gaussian_{j}')
  actions_batch=torch.stack(pool);scores={};predictions={}
  for name,spec in VALUE_ABLATIONS.items():
   with torch.random.fork_rng(devices=cuda):
    pair=evaluator.new_mppi_q_pair() if spec['q_aggregation']=='fixed_pair' else None
    pred=evaluator.evaluate_mppi_candidates(z,actions_batch,compute_cost=False,terminal_action=spec['terminal_action'],q_aggregation=spec['q_aggregation'],q_indices=pair,capture_ensemble=True)
   predictions[name]=pred
   member_j=pred.discounted_reward[None]+pred.terminal_discount[None]*pred.q_member_values
   values=pred.total_value.cpu().numpy();ii=indices[name];best=max(ii['elites'],key=lambda i:values[i]);m=ii['mean']
   scores[name]=dict(values=values.tolist(),prefix_reward=pred.discounted_reward.cpu().tolist(),terminal_q=pred.terminal_q.cpu().tolist(),
    terminal_discount=pred.terminal_discount.cpu().tolist(),terminal_actions=pred.terminal_actions.cpu().tolist(),q_members=pred.q_member_values.cpu().tolist(),
    fixed_pair=pair,ranking=ranking_audit(member_j.cpu().numpy()),
    same_batch_audit=dict(baseline=float(values[0]),mean=float(values[m]),best_recorded_elite=float(values[best]),best_recorded_elite_index=best,
     mean_beats_best_recorded_elite=bool(values[m]>values[best]),mean_beats_baseline=bool(values[m]>values[0])))
  for name,pred in predictions.items():
   assert torch.equal(pred.discounted_reward,predictions['D0'].discounted_reward)
   assert torch.equal(pred.terminal_latent,predictions['D0'].terminal_latent)
   assert torch.equal(pred.terminal_discount,predictions['D0'].terminal_discount)
  assert scores['D0']['values']==scores['D1']['values'] # temperature affects proposals, never the scoring function
  vehicle=env.unwrapped_metadrive.agent;obstacles=agent.planner.structured.planner._collect_obstacles(vehicle)
  state=dict(ego=[*map(float,vehicle.position),*map(float,vehicle.velocity),float(vehicle.heading_theta)],
   obstacles=sorted([list(map(float,(*o.position,*o.velocity,o.heading,o.length,o.width))) for o in obstacles]))
  chosen=0 if a.variant=='sac' else indices['D0']['mean'];actions=actions_batch[chosen];gamma=float(agent.tdmpc_agent.discount)
  trace=[];done=False;crash=offroad=False;H=int(cfg.horizon);wm_z=z.clone();latent_errors=[];terminal={};actual_z=None
  while not done and len(trace)<1000:
   i=len(trace);action=actions[i] if i<H else agent.act(obs,t0=False,eval_mode=True)
   if i==H:terminal['actual_sac_continuation_action']=action.cpu().tolist()
   if i<H:wm_z=agent.tdmpc_agent.model.next(wm_z,action[None],None)
   obs,reward,done,info=env.step(action)
   if i<H:
    actual_z=evaluator.encode(obs);latent_errors.append(float((wm_z-actual_z).square().mean().sqrt()))
   if i+1==H:
    model=agent.tdmpc_agent.model
    terminal['actual_encoded_latent']=actual_z.cpu().tolist();terminal['predicted_latent']=wm_z.cpu().tolist()
    for mode in ('D0','D2'):
     pred=predictions[mode];ta=pred.terminal_actions[chosen:chosen+1]
     qtrue=wm_math.two_hot_inv(model.Q(actual_z,ta,None,return_type='all'),cfg).reshape(-1)
     terminal[mode]=dict(same_terminal_action=ta.cpu().tolist(),predicted_Q_members=pred.q_member_values[:,chosen].cpu().tolist(),actual_state_Q_members=qtrue.cpu().tolist())
    with torch.random.fork_rng(devices=cuda):_,pinfo=model.pi(actual_z,None)
    actual_prior=pinfo['mean'];terminal['actual_prior_action']=actual_prior.cpu().tolist()
    terminal['actual_state_actual_prior_Q_members']=wm_math.two_hot_inv(model.Q(actual_z,actual_prior,None,return_type='all'),cfg).reshape(-1).cpu().tolist()
   crash|=any(bool(info.get(k,False)) for k in ('crash','crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk'))
   offroad|=bool(info.get('out_of_road',False))
   trace.append(dict(action=action.cpu().tolist(),reward=float(reward),cost=float(info.get('cost',0.))))
  assert done,'Probe must continue to native termination'
  if 'actual_sac_continuation_action' in terminal:
   ta=torch.tensor([terminal['actual_sac_continuation_action']],device=device)
   terminal['actual_state_actual_sac_Q_members']=wm_math.two_hot_inv(agent.tdmpc_agent.model.Q(actual_z,ta,None,return_type='all'),cfg).reshape(-1).cpu().tolist()
  actual_prefix=sum(gamma**j*r['reward'] for j,r in enumerate(trace[:H]));rtg=sum(gamma**j*r['reward'] for j,r in enumerate(trace[H:]));ret=sum(gamma**j*r['reward'] for j,r in enumerate(trace))
  assert abs(ret-actual_prefix-gamma**H*rtg)<1e-7
  result=dict(scenario=a.scenario,before=a.before,variant=a.variant,observation=initial_obs,observed_state=state,
   observed_state_id=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest(),replay_max_speed_error=replay_error,
   prefix_source='archived_controls' if archived_origin else 'common_SAC_prefix',prefix_controls=[r['executed_action'] for r in prefix],
   candidate_labels=labels,candidate_actions=actions_batch.cpu().tolist(),variant_indices=indices,search=results,common_pool_scores=scores,
   baseline=baseline.cpu().tolist(),mean=actions_batch[indices['D0']['mean']].cpu().tolist(),chosen_index=chosen,
   actual_first_H_discounted_reward=actual_prefix,actual_terminal_continuation=rtg,actual_discounted_return=ret,
   latent_rollout_rmse=latent_errors,terminal=terminal,gamma=gamma,episode_complete=bool(done),controls=len(trace),success=bool(info.get('success',False)),
   route_completion=float(info.get('route_completion',0.)),crash=crash,offroad=offroad,trace=trace,checkpoint_sha256=meta['checkpoint_sha256'],
   scope='Exactly H candidate commands then SAC continuation to native termination. Common observed state/candidates checked across fresh processes, not a hidden simulator snapshot claim. Latent RMSE is dimensionless. True Q vs SAC RTG contains continuation-policy mismatch. Offline audit only; no runtime gate. Seven additional origins are prespecified SAC-prefix step100, not reconstructed historical failure states.')
  a.output.write_text(json.dumps(result,indent=2));print(a.output,flush=True)
 finally:env.close()
if __name__=='__main__':main()
