"""Native full-episode A/B/C rolling control; A also supplies fixed pre50 H1 origins."""
import argparse,os,sys,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['A','B','C'],required=True);ap.add_argument('--scene',type=int,required=True);ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
project=Path(__file__).resolve().parents[2]
os.environ['MPR_CODE_ROOT']=str(project if args.group=='C' else project.parent/'mpr_mpc_v2')
from study import *

@torch.no_grad()
def main():
    p=load(R/'protocol.json');out=original_dir(args.scene,args.seed) if args.group=='A' else R/'closed'/f'{args.group}_{args.scene}_r{args.seed}'
    out.mkdir(parents=True,exist_ok=False)
    parsed,cfg,env,agent=make_runtime(args.scene,args.seed,{'A':'baseline','B':'native','C':'modified'}[args.group],record=False)
    from common.seed import set_seed
    from omegaconf import OmegaConf
    wm=tensor_hash(agent.model);actor=tensor_hash(agent.residual_sac.actor)
    try:
        combined=args.scene+args.seed;set_seed(combined);agent.planner.reset(seed=combined);obs=env.reset(seed=args.scene)
        assert env.unwrapped_metadrive.config['decision_repeat']*env.unwrapped_metadrive.config['physics_world_step_size']==.1
        initial=snapshot(env,obs);dump(out/'initial.json',initial);OmegaConf.save(parsed,out/'config.yaml')
        local_wall=[0.];original_local=agent.planner.local_mppi.plan
        def local_timed(*args,**kwargs):
            torch.cuda.synchronize();begin=time.perf_counter();result=original_local(*args,**kwargs);torch.cuda.synchronize()
            local_wall[0]=1000*(time.perf_counter()-begin);return result
        agent.planner.local_mppi.plan=local_timed
        observations=[obs.cpu().numpy().copy()];actions=[];rewards=[];costs=[];dones=[];terms=[];truncs=[];physical=[];rows=[];done=crash=oor=False
        while not done:
            step=len(actions)+1;local_wall[0]=0.;torch.cuda.synchronize();t=time.perf_counter();action=agent.act(obs,t0=step==1,eval_mode=True);torch.cuda.synchronize();planning=1000*(time.perf_counter()-t)
            plan=agent.last_plan;assert plan.stage.use_residual and plan.stage.use_mppi==(args.group!='A')
            if step==p['origin_step'] and args.group=='A':
                dump(out/'origin.json',snapshot(env,obs));dump(out/'policy.json',policy_snapshot(agent));dump(out/'environment.json',extended_environment(env))
                np.savez_compressed(out/'origin.npz',refined_actions=plan.refined_actions.cpu().numpy(),latent=agent.planner.last_context.latent.cpu().numpy(),state535=agent.planner.last_context.residual_state.cpu().numpy())
            assert torch.equal(action,plan.debug['selected_mppi_actions'][0])
            first_delta=float((action.cpu()-plan.refined_actions[0].cpu()).norm())
            env_start=time.perf_counter();obs,reward,done,info=env.step(action);env_ms=1000*(time.perf_counter()-env_start)
            assert abs(float(reward)-float(info['tdmpc2_reward']))<1e-4
            vehicle=env.unwrapped_metadrive.agent;received=np.array([float(vehicle.steering),float(vehicle.throttle_brake)])
            assert np.array_equal(received,action.cpu().numpy())
            rows.append(dict(configuration=args.group,scene=args.scene,seed=args.seed,step=step,planner_ms=planning,env_ms=env_ms,control_cycle_ms=planning+env_ms,
                local_mppi_ms=local_wall[0],
                action_delta=first_delta,physical_replacement=first_delta>1e-5,baseline_selected=float(plan.metrics['mpr/baseline_selected']),
                reward=float(info['tdmpc2_reward']),cost=float(info['cost']),route_completion=float(info['route_completion']),
                **{k:float(v) for k,v in plan.metrics.items() if k.startswith('mppi/') or k.startswith('mpr/sac_inference')},
                selected_negative_rate=None,candidate_positive_rate=None,counterfactual_labels='Not observed per online step; use separate H1 branches'))
            observations.append(obs.cpu().numpy().copy());actions.append(action.cpu().numpy().copy());rewards.append(float(info['tdmpc2_reward']));costs.append(float(info['cost']))
            dones.append(bool(done));terms.append(bool(info['terminated']));truncs.append(bool(info['truncated']))
            physical.append([*map(float,vehicle.position[:2]),float(vehicle.speed),float(vehicle.heading_theta)])
            crash|=any(bool(info.get(k,False)) for k in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk']);oor|=bool(info.get('out_of_road',False))
            assert step<=1000
            if step%200==0:print('CLOSED_PROGRESS',args.group,args.scene,args.seed,step,round(info['route_completion'],3),flush=True)
        assert wm==tensor_hash(agent.model) and actor==tensor_hash(agent.residual_sac.actor)
        np.savez_compressed(out/'trace.npz',observations=observations,actions=actions,rewards=rewards,costs=costs,done=dones,terminated=terms,truncated=truncs,physical_after=physical)
        table(out/'steps.csv',rows)
        dump(out/'episode.json',dict(configuration=args.group,scene=args.scene,seed=args.seed,success=float(info['success']),crash=float(crash),out_of_road=float(oor),route_completion=float(info['route_completion']),
            timeout=float(info['truncated']),episode_return=sum(rewards),discounted_return=discounted_return(rewards),episode_length=len(rewards),safety_cost=sum(costs),
            replacement_rate=float(np.mean([x['physical_replacement'] for x in rows])),mean_action_deviation=float(np.mean([x['action_delta'] for x in rows])),
            planning_p50=float(np.quantile([x['planner_ms'] for x in rows],.5)),planning_p95=float(np.quantile([x['planner_ms'] for x in rows],.95)),
            cycle_p95=float(np.quantile([x['control_cycle_ms'] for x in rows],.95)),initial_hashes=initial['hashes'],end_reason=info.get('end_reason',''),
            native_done=True,weights_unchanged=True,reachable_h1_origin=(out/'origin.json').exists() if args.group=='A' else None,
            latency_scope='6worker shared GPU native episode, cold start retained; separate isolated probe for real-time assessment'))
        print('CLOSED_DONE',args.group,args.scene,args.seed,'length',len(actions),'return',round(discounted_return(rewards),3),flush=True)
    finally:env.close()
if __name__=='__main__':main()
