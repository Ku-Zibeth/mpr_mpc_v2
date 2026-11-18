"""Physical H=1/2/4/10 intervention followed by ORIGINAL feedbackLattice+SAC."""
import argparse,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *
from tools.policy_value_diagnosis import discounted_return

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True);ap.add_argument('--before',type=int,required=True);ap.add_argument('--candidate',required=True);ap.add_argument('--horizon',type=int,required=True)
    ap.add_argument('--takeover',choices=['baseline','td_actor','td_mpc'],default='baseline');ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    p=load(REPORT/'protocol.json');assert a.horizon in p['horizons'];origin=REPORT/'origins'/f'{a.scene}_{a.before}';catalog=np.load(origin/'catalog.npz');meta=load(origin/'catalog.json')
    idx=next(i for i,r in enumerate(meta['representatives']) if r['candidate_id']==a.candidate);commands=torch.tensor(catalog['actions'][idx],device='cuda')
    label=f'H{a.horizon}_{a.candidate}_{a.takeover}'+('_replay' if a.replay else '')
    out=REPORT/'branches'/origin.name;out.mkdir(parents=True,exist_ok=True);target=out/(label+'.npz');assert not target.exists()
    cfg,env,agent=make_runtime(a.scene);initial_hash=tensor_hash(agent.model);actor_hash=tensor_hash(agent.residual_sac.actor)
    try:
        obs,unreachable=at_origin(env,agent,a.scene,a.before);assert unreachable is None
        assert np.array_equal(agent.last_plan.refined_actions.cpu().numpy(),catalog['baseline'])
        assert np.array_equal(agent.planner.last_context.residual_state.cpu().numpy(),catalog['state535'])
        torch.random.set_rng_state(torch.tensor(catalog['post_cpu_rng'],dtype=torch.uint8));torch.cuda.set_rng_state(torch.tensor(catalog['post_cuda_rng'],dtype=torch.uint8))
        start=snapshot(env,obs);assert_pair(start,load(origin/'origin.json'));dump(out/(label+'_origin.json'),dict(hashes=start['hashes'],rng_component_hashes=start['rng_component_hashes']))
        observations=[obs.cpu().numpy().copy()];actions=[];rewards=[];costs=[];dones=[];terminated=[];truncated=[];invalid=[];planning=[];done=crash=offroad=False;feedback=0;first_sac_postprefix=None
        while not done:
            t=len(actions);bad=False;ms=0.
            if t<a.horizon:
                action=commands[t].cpu()  # already PHYSICAL; never run throughresidual adapter!
            elif a.takeover=='baseline':
                begin=time.perf_counter();action=agent.act(obs,t0=False,eval_mode=True);ms=1000*(time.perf_counter()-begin)
                assert_feedback_stage(agent.last_plan.stage)
                feedback+=1;bad=not bool(agent.last_plan.residual_valid)
                if first_sac_postprefix is None:first_sac_postprefix=action.cpu().numpy().copy()
            elif a.takeover=='td_actor':
                z=agent.planner.evaluator.encode(obs);action=agent.model.pi(z,None)[0][0].cpu()
            else:
                assert cfg.mpc and int(cfg.num_samples)==512 and int(cfg.num_elites)==64 and int(cfg.iterations)==6
                begin=time.perf_counter();action=agent.tdmpc_agent.act(obs,t0=t==a.horizon,eval_mode=True);ms=1000*(time.perf_counter()-begin)
            obs,reward,done,info=env.step(action);assert abs(float(reward)-float(info['tdmpc2_reward']))<1e-4
            observations.append(obs.cpu().numpy().copy());actions.append(action.cpu().numpy().copy());rewards.append(float(info['tdmpc2_reward']));costs.append(float(info['cost']));dones.append(bool(done));terminated.append(bool(info['terminated']));truncated.append(bool(info['truncated']));invalid.append(bad);planning.append(ms)
            crash|=any(bool(info.get(k,False)) for k in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk']);offroad|=bool(info.get('out_of_road',False));assert t<1000
        assert np.array_equal(actions[0],physical_first_action(catalog['actions'][idx]))
        if a.takeover=='baseline':assert feedback==max(0,len(actions)-a.horizon)
        rtg=np.zeros(len(rewards)+1)
        for t in range(len(rewards)-1,-1,-1):rtg[t]=rewards[t]+.995*rtg[t+1]
        assert abs(rtg[0]-discounted_return(rewards,.995,dones))<1e-8
        np.savez_compressed(target,observations=observations,actions=actions,rewards=rewards,costs=costs,done=dones,terminated=terminated,truncated=truncated,rtg=rtg,
            invalid_residual=invalid,planning_ms=planning)
        row=dict(scenario_id=a.scene,scenario=a.scene,origin_id=origin.name,episode_seed=a.scene,policy_seed=1+a.scene,candidate_id=a.candidate,horizon=a.horizon,takeover=a.takeover,replay=a.replay,
            origin_observation_hash=start['hashes']['observation'],origin_vehicle_state_hash=start['hashes']['vehicle'],origin_traffic_state_hash=start['hashes']['traffic'],origin_rng_state_hash=start['hashes']['rng'],
            candidate_first_action=catalog['actions'][idx,0],baseline_first_action=catalog['baseline'][0],actual_first_reward=rewards[0],actual_continuation_return=float(rtg[1]),
            actual_return=float(rtg[0]),episode_return=sum(rewards),success=float(info['success']),crash=float(crash),out_of_road=float(offroad),route_completion=float(info['route_completion']),safety_cost=sum(costs),
            invalid_residual_count=sum(invalid),episode_length=len(actions),termination_reason=info.get('end_reason',''),native_terminated=terminated[-1],native_truncated=truncated[-1],native_done=dones[-1],
            feedback_lattice_sac_steps=feedback,first_postprefix_sac_action=first_sac_postprefix,physical_action_mapping_count=0,weights_unchanged=initial_hash==tensor_hash(agent.model) and actor_hash==tensor_hash(agent.residual_sac.actor),
            protocol='Y1=firstreward+gamma*feedbackSACreturn' if a.horizon==1 and a.takeover=='baseline' else 'H-prefixthenfeedbackcontinuation',
            source=str(target.relative_to(ROOT)))
        assert row['weights_unchanged'];dump(target.with_suffix('.json'),row);print('DONE',origin.name,label,'return',round(rtg[0],3),'success',row['success'],flush=True)
    finally:env.close()

if __name__=='__main__':main()
