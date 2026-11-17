"""Reconstruct existing Gaussian pools before observing new real suffix rewards."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from diagnosis_runtime import make,at_origin,snapshot,dump
import numpy as np,torch
from planning.local_mppi import LocalMPPI
from planning.mppi_ablation import iterations_ablation_config

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scenario',type=int,required=True);ap.add_argument('--before',type=int,required=True);a=ap.parse_args();p,cfg,env,agent=make(a.scenario);assert [a.scenario,a.before] in p['development_origins'];folder=R/'catalog'/f'{a.scenario}_{a.before}';folder.mkdir(parents=True,exist_ok=True);assert not (folder/'catalog.npz').exists()
 try:
  obs,ctx,baseline=at_origin(env,agent,a.scenario,a.before);pre_cpu=torch.random.get_rng_state().clone();pre_cuda=torch.cuda.get_rng_state().clone();before=snapshot(env,agent,obs,baseline[None],a.scenario,a.before)
  evaluator=agent.planner.evaluator;model=agent.tdmpc_agent.model;oldq=model.Q;calls=[];pairs=[];old_eval=evaluator.evaluate_mppi_candidates
  def q(z,act,task,return_type='min',**kw):
   if return_type=='avg':
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):pair=torch.randperm(agent.tdmpc_agent.cfg.num_q,device=agent.device)[:2].cpu().tolist()
    pairs.append(pair)
   return oldq(z,act,task,return_type=return_type,**kw)
  def capture(z,actions,**kw):
   value=old_eval(z,actions,**kw);calls.append((actions.clone(),value));return value
  model.Q=q;evaluator.evaluate_mppi_candidates=capture
  options=dict(cfg.mpr_mpc['mppi']);options.update(audit_candidates=True,record_cost=False,cost_weight=0.)
  planner=LocalMPPI(evaluator,options,horizon=10,action_dim=2,device=agent.device);planner.reset(1+a.scenario);plan=planner.plan(ctx.latent,baseline,eval_mode=True)
  post_cpu=torch.random.get_rng_state().clone();post_cuda=torch.cuda.get_rng_state().clone();model.Q=oldq;evaluator.evaluate_mppi_candidates=old_eval
  prior=Path(p['prior_data'])/'data'/f'{a.scenario}_{a.before}';saved=np.load(prior/'audit.npz');assert np.array_equal(torch.stack(plan.candidate_audit['iteration_actions']).cpu().numpy(),saved['iteration_actions']);assert np.array_equal(torch.stack(plan.candidate_audit['iteration_values']).cpu().numpy(),saved['iteration_scores'])
  assert np.array_equal(plan.candidate_audit['actions'].cpu().numpy(),saved['actions'])
  old_origin=json.loads((prior/'baseline_origin.json').read_text());post=snapshot(env,agent,obs,plan.candidate_audit['actions'][[0,2,1,plan.candidate_audit['indices']['best_elite']]],a.scenario,a.before)
  for key in ['observation','state','rng','episode_config','candidate_actions']:assert post[key]==old_origin[key],key
  actions=[baseline,plan.candidate_audit['actions'][2]];native_records=[]
  for i in range(4):
   aa,result=calls[i];values=result.total_value;best=int(values.argmax());actions.extend([torch.tensor(plan.iterations[i]['mean'],device=agent.device),aa[best]])
   native_records.append(dict(k=i+1,raw_pair=pairs[i],raw_best_gain=float(values[best]-values[0]),best_pool_index=best,top8=torch.topk(values,8).indices.cpu().tolist(),raw_baseline=float(values[0]),raw_best=float(values[best])))
  pool=torch.stack(actions);refpair=pairs[-1];consequence=evaluator.evaluate_mppi_candidates(ctx.latent,pool,compute_cost=False,q_aggregation='fixed_pair',q_indices=refpair)
  iterations={}
  for k in p['iterations_ablation']:
   with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
    torch.random.set_rng_state(pre_cpu);torch.cuda.set_rng_state(pre_cuda);op=iterations_ablation_config(cfg.mpr_mpc['mppi'],k);op.update(audit_candidates=False,record_cost=False,cost_weight=0.);one=LocalMPPI(evaluator,op,horizon=10,action_dim=2,device=agent.device);one.reset(1+a.scenario);result=one.plan(ctx.latent,baseline,eval_mode=True)
   assert torch.equal(result.final_mean,pool[2+2*(k-1)])
   label='baseline' if result.baseline_selected else f'mean{k}'
   iterations[k]=dict(selected=label,predicted_gain=float(result.selected_wm_value-result.baseline_wm_value),raw_mean_gain=float(result.optimized_wm_value-result.baseline_wm_value),evaluation=result.evaluation)
  z=ctx.latent.expand(len(pool),-1).clone();zs=[z.cpu().numpy()]
  for h in range(10):z=model.next(z,pool[:,h],None);zs.append(z.cpu().numpy())
  assert ctx.safety_context is not None;xy,yaw,speed=agent.planner.safety.rollout(pool,ctx.safety_context);stride=agent.planner.safety.substeps
  np.savez_compressed(folder/'catalog.npz',actions=pool.cpu().numpy(),z0=ctx.latent.cpu().numpy().reshape(-1),predicted_latents=np.stack(zs,1),predicted_values=consequence.total_value.cpu().numpy(),predicted_prefix=consequence.discounted_reward.cpu().numpy(),ref_pair=refpair,
   physics_xy=xy[:,::stride].cpu().numpy(),physics_yaw=yaw[:,::stride].cpu().numpy(),physics_speed=speed[:,::stride].cpu().numpy(),post_cpu_rng=post_cpu.numpy(),post_cuda_rng=post_cuda.cpu().numpy())
  dump(folder/'pre_origin.json',before);torch.random.set_rng_state(post_cpu);torch.cuda.set_rng_state(post_cuda);dump(folder/'origin.json',snapshot(env,agent,obs,pool,a.scenario,a.before));dump(folder/'catalog.json',dict(scenario=a.scenario,before=a.before,choices=p['choices'],iterations=iterations,raw_rounds=native_records,common_q_pair=refpair,exact_existing_pool_reconstruction=True,budget=options,online_default_unchanged=True))
  print(a.scenario,a.before,'catalog prepared, cached candidates identical',flush=True)
 finally:env.close()
if __name__=='__main__':main()
