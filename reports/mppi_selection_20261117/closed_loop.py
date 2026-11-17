"""Native terminal episodes; Lattice/SAC/MPPI recomputed on EVERY control tick."""
import argparse,contextlib,io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'))
from probe import dump,snapshot
import numpy as np,torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite
from planning.sac_value import policy_context_sha256


def method(name,p):
 if name in p['stage1']:return p['stage1'][name]
 k=json.loads((R/'iteration_choice.json').read_text())['iterations'];options=dict(mppi=True,iterations=k,terminal_action='policy_prior' if name in ['P','PE'] else 'candidate',selection_mode='best_final_elite' if name in ['E','PE'] else 'weighted_mean')
 assert name in ['P','E','PE'];return options


@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','validation'],required=True);ap.add_argument('--group',required=True);ap.add_argument('--scenario',type=int,required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();p=json.loads((R/'protocol.json').read_text());assert a.scenario in p[a.phase+'_scenarios'];assert a.seed in p['repeat_seeds'];opt=method(a.group,p);ck=Path(p['checkpoint']);out=R/'episodes'/a.phase/f'{a.group}_{a.scenario}_r{a.seed}';out.mkdir(parents=True,exist_ok=False)
 raw=OmegaConf.load(p['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi;raw.mpr_mpc.mppi.update({k:v for k,v in opt.items() if k!='mppi'});raw.mpr_mpc.mppi.update(dict(audit_candidates=False,record_candidates=True,record_cost=False,cost_weight=0.))
 raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8;raw.metadrive.simulator.horizon=1000;raw.seed=a.seed
 parsed,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[a.scenario]);assert cfg.horizon==10;OmegaConf.save(parsed,out/'config.yaml');torch.set_num_threads(1);torch.set_float32_matmul_precision('high');set_seed(a.seed);env=make_env(cfg)
 try:
  with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);state=agent.load(ck)
  agent.eval();suite._apply_evaluation_stack(agent.planner,'full_mpr_mpc' if opt['mppi'] else 'world_model_residual_rl',state);agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step));assert agent.planner.residual_state_dim==535
  seed=a.seed+a.scenario;set_seed(seed);agent.planner.reset(seed=seed);obs=env.reset(seed=a.scenario);assert int(env.unwrapped_metadrive.current_seed)==a.scenario
  initial=snapshot(env,agent,obs,torch.empty((0,11,2),device=agent.device),a.scenario,1);initial['episode_config']['seed']=seed;initial['episode_config']['repeat_seed']=a.seed;dump(out/'origin.json',initial)
  context_reference=ROOT/'reports/sac_value_20261117/dataset_manifest.json';assert policy_context_sha256(cfg)==json.loads(context_reference.read_text())['context']
  counts=dict(lattice_prepare=0,mppi=0);prepare=agent.planner.prepare;plan=agent.planner.local_mppi.plan
  def wrapped_prepare(*args,**kw):counts['lattice_prepare']+=1;return prepare(*args,**kw)
  def wrapped_plan(*args,**kw):counts['mppi']+=1;return plan(*args,**kw)
  agent.planner.prepare=wrapped_prepare;agent.planner.local_mppi.plan=wrapped_plan
  observations=[obs.cpu().numpy().copy()];actions=[];rewards=[];costs=[];dones=[];times=[];planner_ms=[];base_selected=[];correction=[];gains=[];sources=[];iteration_actions=[];iteration_values=[];iteration_members=[];final_actions=[];final_values=[];final_members=[];done=crash=offroad=False
  while not done:
   i=len(actions);torch.cuda.synchronize();begin=time.perf_counter();action=agent.act(obs,t0=(i==0),eval_mode=True);torch.cuda.synchronize();ms=1000*(time.perf_counter()-begin);result=agent.last_plan;assert result.stage.use_residual and result.stage.use_mppi==opt['mppi'];assert counts['lattice_prepare']==i+1
   if opt['mppi']:
    assert counts['mppi']==i+1;assert torch.equal(action,result.debug['selected_mppi_actions'][0]);assert result.metrics['mppi/iterations_completed']==opt['iterations'];assert result.metrics['mppi/wm_evaluated_candidates']==opt['iterations']*65+2
    selection=result.debug['mppi_selection_candidates'];trace=result.debug['mppi_candidate_trace'];assert len(trace['actions'])==opt['iterations'];assert len(selection['q_member_indices'])==2
    iteration_actions.append(torch.stack(trace['actions']).cpu().numpy());iteration_values.append(torch.stack(trace['values']).cpu().numpy());iteration_members.append(np.asarray(trace['q_members'],np.int64));final_actions.append(selection['actions'].cpu().numpy());final_values.append(selection['values'].cpu().numpy());final_members.append(np.asarray(selection['q_member_indices'],np.int64))
    source=int(result.metrics['mppi/selected_source_code']);assert source in ([0,1] if opt['selection_mode']=='weighted_mean' else [0,2]);assert torch.equal(action,selection['actions'][source,0].cpu());assert source==0 or float(selection['values'][source])>float(selection['values'][0]);sources.append(source)
    if opt['terminal_action']=='policy_prior':
     assert result.metrics['mppi/active_action_steps']==10;assert (result.debug['final_mppi_std'][-1]==0).all();assert torch.equal(result.debug['final_mppi_mean'][-1],result.refined_actions[-1].cpu())
    base_selected.append(float(result.metrics['mppi/baseline_selected']));correction.append(float(result.metrics['mppi/first_action_delta_norm']));gains.append(float(result.metrics['mppi/score_gain']))
   else:assert counts['mppi']==0;sources.append(0)
   times.append(ms);planner_ms.append(float(result.metrics['total_planning_ms']));obs,_,done,info=env.step(action);observations.append(obs.cpu().numpy().copy());actions.append(action.cpu().numpy().copy());rewards.append(float(info['tdmpc2_reward']));costs.append(float(info['cost']));dones.append(bool(done));crash|=any(bool(info.get(k,False)) for k in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk']);offroad|=bool(info.get('out_of_road',False));assert len(actions)<=1000
   if (i+1)%100==0:print(a.group,a.scenario,a.seed,'step',i+1,'route',info['route_completion'],'act_ms',round(ms,1),flush=True)
  gamma=float(agent.tdmpc_agent.discount);assert gamma==p['gamma'];actual=sum(gamma**i*r for i,r in enumerate(rewards));arr=dict(observations=observations,actions=actions,rewards=rewards,costs=costs,done=dones,act_wall_ms=times,planner_ms=planner_ms,baseline_selected=base_selected,first_correction=correction,selected_gain=gains,selected_source=sources)
  if opt['mppi']:arr.update(iteration_actions=iteration_actions,iteration_values=iteration_values,iteration_q_members=iteration_members,final_candidates=final_actions,final_values=final_values,final_q_members=final_members)
  np.savez_compressed(out/'trace.npz',**arr)
  row=dict(configuration=a.group,scenario=a.scenario,test_seed=a.seed,success=float(info['success']),crash=float(crash),out_of_road=float(offroad),route_completion=float(info['route_completion']),episode_return=sum(rewards),actual_discounted_return=actual,safety_cost=sum(costs),length=len(actions),baseline_selected_rate=float(np.mean(base_selected)) if base_selected else None,first_action_correction=float(np.mean(correction)) if correction else 0.,selected_predicted_gain=float(np.mean(gains)) if gains else None,planning_time_ms=float(np.mean(times)),planning_p95_ms=float(np.quantile(times,.95)),counts=counts,native_done=True,options=opt,policy_context_sha256=policy_context_sha256(cfg),end_reason=info.get('end_reason',''),measurement='Complete agent.act synchronized wall under sharedGPU load; each first action actually executed before new planning. Not isolated latency.')
  dump(out/'episode.json',row);print('DONE',a.group,a.scenario,a.seed,actual,row['success'],row['crash'],row['out_of_road'],flush=True)
 finally:env.close()
if __name__=='__main__':main()
