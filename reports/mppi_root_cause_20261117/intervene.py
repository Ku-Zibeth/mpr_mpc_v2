"""Same stored pool; native H10 or first-action-only intervention then frozen SAC."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from diagnosis_runtime import make,at_origin,snapshot,dump,physical,actor_audit
import numpy as np,torch

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scenario',type=int,required=True);ap.add_argument('--before',type=int,required=True);ap.add_argument('--choice',required=True);ap.add_argument('--protocol',choices=['A','B'],required=True);a=ap.parse_args();p,cfg,env,agent=make(a.scenario);folder=R/'catalog'/f'{a.scenario}_{a.before}';catalog=np.load(folder/'catalog.npz');names=p['choices'];index=names.index(a.choice);commands=torch.tensor(catalog['actions'][index],device=agent.device);out=R/'interventions'/f'{a.scenario}_{a.before}';out.mkdir(parents=True,exist_ok=True);target=out/f'{a.protocol}_{a.choice}.npz';assert not target.exists()
 try:
  obs,context,baseline=at_origin(env,agent,a.scenario,a.before);reference=json.loads((folder/'pre_origin.json').read_text());pre=snapshot(env,agent,obs,baseline[None],a.scenario,a.before)
  for key in ['episode_config','observation','state','rng','candidate_actions']:assert pre[key]==reference[key],key
  torch.random.set_rng_state(torch.tensor(catalog['post_cpu_rng'],dtype=torch.uint8));torch.cuda.set_rng_state(torch.tensor(catalog['post_cuda_rng'],dtype=torch.uint8));pool=torch.tensor(catalog['actions'],device=agent.device);post=snapshot(env,agent,obs,pool,a.scenario,a.before);reference=json.loads((folder/'origin.json').read_text());assert post==reference
  dump(out/f'{a.protocol}_{a.choice}_origin.json',post)
  observations=[obs.cpu().numpy().copy()];states=[physical(env)];actions=[];rewards=[];costs=[];dones=[];done=crash=offroad=False;limit=p['interventions'][a.protocol];hstate=None
  while not done:
   i=len(actions);act=commands[i] if i<limit else agent.act(obs,t0=False,eval_mode=True)
   if i==10:hstate=agent.planner.last_context.residual_state.cpu().numpy().copy()
   obs,_,done,info=env.step(act);observations.append(obs.cpu().numpy().copy());states.append(physical(env));actions.append(act.cpu().numpy().copy());rewards.append(float(info['tdmpc2_reward']));costs.append(float(info['cost']));dones.append(bool(done))
   crash|=any(bool(info.get(k,False)) for k in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk']);offroad|=bool(info.get('out_of_road',False));assert len(actions)<=1000
  gamma=float(agent.tdmpc_agent.discount);rtg=np.zeros(len(rewards)+1)
  for t in range(len(rewards)-1,-1,-1):rtg[t]=rewards[t]+gamma*rtg[t+1]
  observations=np.asarray(observations,np.float32);zs=agent.planner.evaluator.encode(torch.tensor(observations[:11])).cpu().numpy()
  np.savez_compressed(target,observations=observations,actions=actions,rewards=rewards,costs=costs,done=dones,rtg=rtg,physical=states,actual_prefix_latents=zs,state535=hstate if hstate is not None else np.empty(0))
  dump(out/f'{a.protocol}_{a.choice}.json',dict(scenario=a.scenario,before=a.before,choice=a.choice,protocol=a.protocol,intervention_steps=limit,actual_return=float(rtg[0]),episode_return=sum(rewards),cost=sum(costs),length=len(actions),success=bool(info['success']),crash=crash,out_of_road=offroad,route_completion=float(info['route_completion']),predicted_return=float(catalog['predicted_values'][index]),native_done=True))
  print(a.scenario,a.before,a.protocol,a.choice,float(rtg[0]),flush=True)
 finally:env.close()
if __name__=='__main__':main()
