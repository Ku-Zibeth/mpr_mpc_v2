"""Replay only recorded H-prefixes to obtain original real535; never train SAC."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from diagnosis_runtime import make,at_origin,snapshot,dump,physical,actor_audit
import numpy as np,torch
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common import math as td_math

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);a=ap.parse_args();source=Path(a.source);scenario,before=map(int,source.parent.name.split('_'));branch=source.stem;p,cfg,env,agent=make(scenario);assert scenario in p['cv_scenarios'];new=source.is_relative_to(Path(p['prior_data']));out=R/'states'/source.parent.name;out.mkdir(parents=True,exist_ok=True);target=out/(branch+'.npz');assert not target.exists();saved=np.load(source);meta=json.loads(source.with_name(branch+'_meta.json').read_text());reference=json.loads(source.with_name(branch+'_origin.json').read_text())
 try:
  obs,ctx,baseline=at_origin(env,agent,scenario,before);index=(['baseline','gaussian','mean','best_elite'] if new else ['B','gaussian_1','gaussian_2']).index(branch);pool=torch.tensor(reference['candidate_actions'],device=agent.device);commands=pool[index];assert np.array_equal(saved['actions'][:10],commands[:min(10,len(saved['actions']))].cpu().numpy());assert np.array_equal(obs.cpu().numpy(),saved['observations'][0])
  if new:
   for _ in range(6):torch.randperm(agent.tdmpc_agent.cfg.num_q,device=agent.device)
  replay=snapshot(env,agent,obs,pool,scenario,before)
  for key in ['observation','state','rng','episode_config','candidate_actions']:assert replay[key]==reference[key],key
  z=ctx.latent.clone();pred=[z.cpu().numpy().reshape(-1)];actual=[z.cpu().numpy().reshape(-1)];physical_states=[physical(env)];done=False;observations=[obs.cpu().numpy().copy()]
  assert ctx.safety_context is not None;xy,yaw,speed=agent.planner.safety.rollout(commands[None],ctx.safety_context);stride=agent.planner.safety.substeps
  for h in range(min(10,len(saved['actions']))):
   z=agent.tdmpc_agent.model.next(z,commands[h:h+1],None);pred.append(z.cpu().numpy().reshape(-1));obs,_,done,info=env.step(commands[h]);assert np.array_equal(obs.cpu().numpy(),saved['observations'][h+1]);actual.append(agent.planner.evaluator.encode(obs).cpu().numpy().reshape(-1));physical_states.append(physical(env));observations.append(obs.cpu().numpy().copy())
   assert abs(float(info['tdmpc2_reward'])-saved['rewards'][h])<1e-8
   if done:break
  count=len(actual)-1;state535=np.empty(0);original_action=np.empty(0);effects={}
  if count==10 and not done:
   agent.act(obs,t0=False,eval_mode=True);context=agent.planner.last_context;state535=context.residual_state;assert tuple(state535.shape)==(535,);original_action,effects=actor_audit(agent,state535);state535=state535.cpu().numpy()
  else:state535=np.empty(0)
  own_pred=[];own_real=[];common_pred=[];common_real=[]
  for h in p['horizons']:
   if h>count:
    own_pred.append(np.full(5,np.nan));own_real.append(np.full(5,np.nan));common_pred.append(np.full(5,np.nan));common_real.append(np.full(5,np.nan));continue
   for latent,controls,result in [(pred[h],commands[h],own_pred),(actual[h],commands[h],own_real),(pred[h],pool[0,h],common_pred),(actual[h],pool[0,h],common_real)]:
    zz=torch.tensor(latent,device=agent.device).reshape(1,-1);aa=controls.reshape(1,-1);q=td_math.two_hot_inv(agent.tdmpc_agent.model.Q(zz,aa,None,return_type='all'),agent.tdmpc_agent.cfg).reshape(-1).cpu().numpy();result.append(q)
  np.savez_compressed(target,predicted=np.asarray(pred),actual=np.asarray(actual),actual_physical=np.asarray(physical_states),kinematic_xy=xy[0,::stride].cpu().numpy(),kinematic_yaw=yaw[0,::stride].cpu().numpy(),kinematic_speed=speed[0,::stride].cpu().numpy(),state535=state535,original_actor_action=original_action,actor_effects=np.stack([effects.get(n,np.full(2,np.nan)) for n in ['path','wm_return','feasibility','risk']]),commands=commands.cpu().numpy(),q_pred_own=np.asarray(own_pred),q_real_own=np.asarray(own_real),q_pred_common=np.asarray(common_pred),q_real_common=np.asarray(common_real),source=str(source),actual_continuation=np.array(float(saved['rtg'][10]) if count==10 else np.nan),actual_return=np.array(float(saved['rtg'][0])),actual_prefix=np.array(sum(p['gamma']**i*x for i,x in enumerate(saved['rewards'][:10]))))
  dump(out/(branch+'.json'),dict(scenario=scenario,before=before,branch=branch,source=str(source),h_available=count,full_state_available=len(state535)==535,native_early_termination=done,prefix_exactly_replayed=True,real_state_builder='original agent.planner.last_context.residual_state from actual H observation',risk_feature_order=agent.planner.residual_state_builder.risk_features,coverage=reference['coverage'],end=meta))
  print(scenario,before,branch,'state535',len(state535),'exact replay',flush=True)
 finally:env.close()
if __name__=='__main__':main()
