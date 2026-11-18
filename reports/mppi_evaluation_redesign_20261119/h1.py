"""Exactly one physical control intervention, then frozen original Lattice+SAC until native done."""
import argparse,time
from study import *

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--candidate',required=True);a=ap.parse_args();a.step=50
    directory=origin_dir(a.scene,a.seed);catalog=load(directory/'catalog.json')
    candidate=next(x for x in catalog['representatives'] if x['candidate_id']==a.candidate)
    commands=np.asarray(candidate['action'],np.float32)
    out=R/'branches'/f'{a.scene}_r{a.seed}'/a.candidate;out.mkdir(parents=True,exist_ok=False)
    parsed,cfg,env,agent=make_runtime(a.scene,a.seed)
    before_wm=tensor_hash(agent.model);before_actor=tensor_hash(agent.residual_sac.actor)
    try:
        obs,baseline,source,state,policy,_=restore_origin(env,agent,a.scene,a.seed)
        assert state['hashes']==load(directory/'origin.json')['hashes']
        assert policy['hashes']==catalog['policy_fingerprint'],('policy history mismatch',a.scene,a.step)
        environment=extended_environment(env);assert environment['hash']==load(directory/'environment.json')['hash']
        assert agent._pending_residual is None
        dump(out/'origin_verification.json',dict(physical_rng_hashes=state['hashes'],policy_hashes=policy['hashes'],
            extended_environment_hash=environment['hash'],all_exact=True,previous_recorded_controls=a.step-1))
        vehicle=env.unwrapped_metadrive.agent
        initial_position=np.asarray(vehicle.position[:2]).copy();initial_speed=float(vehicle.speed);initial_heading=float(vehicle.heading_theta)
        first_immediate=None;steps=[];observations=[obs.cpu().numpy().copy()];controls=[];rewards=[];costs=[];dones=[];terms=[];truncs=[];physical=[]
        done=crash=offroad=False;feedback=0;previous=None;baseline_match_count=0
        def forbidden_mppi(*args,**kwargs):raise AssertionError('MPPI must never run inside a continuation branch')
        agent.planner.local_mppi.plan=forbidden_mppi
        native_prev_mean=agent.tdmpc_agent._prev_mean.clone()
        while not done:
            k=len(controls);before=time.perf_counter()
            if k==0:
                planned=baseline.cpu().numpy();action=commands.copy();takeover=False
            else:
                planned=agent.act(obs,t0=False,eval_mode=True).cpu().numpy();action=planned.copy();takeover=True;feedback+=1
                assert agent.last_plan.stage.use_residual and not agent.last_plan.stage.use_mppi
                assert agent._pending_residual is None
                ctx=agent.planner.last_context
                if ctx.safety_context is not None:
                    assert np.array_equal(ctx.safety_context.previous_action,previous),'Previous actual action not reflected in safety/SAC feature context'
                if k==1:
                    assert torch.equal(ctx.latent,agent.planner.evaluator.encode(obs).detach())
                    assert torch.equal(ctx.residual_state[:512],ctx.latent.reshape(-1))
            planning=1000*(time.perf_counter()-before)
            assert np.isfinite(action).all() and np.max(np.abs(action))<=1
            obs,reward,done,info=env.step(action)
            assert abs(float(reward)-float(info['tdmpc2_reward']))<1e-4
            vehicle=env.unwrapped_metadrive.agent;received=np.array([float(vehicle.steering),float(vehicle.throttle_brake)])
            assert np.array_equal(received,action),'Submitted and actual MetaDrive control differ'
            previous=action.copy();position=np.asarray(vehicle.position[:2]).copy();speed=float(vehicle.speed);heading=float(vehicle.heading_theta)
            body=np.r_[position,speed,heading]
            if a.candidate=='A':
                original=a.step-1+k
                assert np.array_equal(action,source['actions'][original]),('Baseline action mismatch',original)
                assert np.array_equal(obs.cpu().numpy(),source['observations'][original+1]),('Baseline observation mismatch',original)
                assert np.array_equal(body,source['physical_after'][original]),('Baseline physical mismatch',original)
                assert abs(float(info['tdmpc2_reward'])-float(source['rewards'][original]))<1e-8
                assert bool(done)==bool(source['done'][original]) and bool(info['terminated'])==bool(source['terminated'][original]) and bool(info['truncated'])==bool(source['truncated'][original])
                baseline_match_count+=1
            if k==0:first_immediate=dict(reward=float(info['tdmpc2_reward']),cost=float(info['cost']),speed_change=speed-initial_speed,
                heading_change=float((heading-initial_heading+np.pi)%(2*np.pi)-np.pi),position_change=float(np.linalg.norm(position-initial_position)))
            steps.append(dict(scene=a.scene,seed=a.seed,origin_step=a.step,candidate_id=a.candidate,branch_step=k,absolute_step=a.step+k,
                executed_steering=float(action[0]),executed_throttle_brake=float(action[1]),received_steering=float(received[0]),received_throttle_brake=float(received[1]),
                planner_steering=float(planned[0]),planner_throttle_brake=float(planned[1]),sac_takeover=takeover,
                intervention=k==0,previous_action_context_verified=k>0,reward=float(info['tdmpc2_reward']),safety_cost=float(info['cost']),
                x=float(position[0]),y=float(position[1]),speed=speed,heading=heading,
                route_completion=float(info['route_completion']),crash=bool(info.get('crash',False)),out_of_road=bool(info.get('out_of_road',False)),
                terminated=bool(info['terminated']),truncated=bool(info['truncated']),done=bool(done),planner_ms=planning))
            observations.append(obs.cpu().numpy().copy());controls.append(action);rewards.append(float(info['tdmpc2_reward']));costs.append(float(info['cost']))
            dones.append(bool(done));terms.append(bool(info['terminated']));truncs.append(bool(info['truncated']));physical.append(body)
            crash|=any(bool(info.get(key,False)) for key in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk'])
            offroad|=bool(info.get('out_of_road',False));assert a.step+k<=1000
            if k and k%200==0:print('PROGRESS',a.scene,a.step,a.candidate,k,round(info['route_completion'],3),flush=True)
        assert feedback==len(controls)-1 and sum(x['intervention'] for x in steps)==1
        assert torch.equal(native_prev_mean,agent.tdmpc_agent._prev_mean)
        assert before_wm==tensor_hash(agent.model) and before_actor==tensor_hash(agent.residual_sac.actor)
        if a.candidate=='A':assert baseline_match_count==len(source['actions'])-a.step+1
        np.savez_compressed(out/'trace.npz',observations=observations,actions=controls,rewards=rewards,costs=costs,
            done=dones,terminated=terms,truncated=truncs,physical_after=physical)
        table(out/'steps.csv',steps)
        row=dict(scene=a.scene,seed=a.seed,origin_step=a.step,candidate_id=a.candidate,roles=candidate['roles'],normalized_action=commands,
            success=float(info['success']),crash=float(crash),out_of_road=float(offroad),route_completion=float(info['route_completion']),
            timeout=float(info['truncated']),safety_cost=sum(costs),episode_length=len(rewards),episode_return=sum(rewards),discounted_return=discounted_return(rewards),
            immediate=first_immediate,end_reason=info.get('end_reason',''),native_terminated=terms[-1],native_truncated=truncs[-1],native_done=dones[-1],
            baseline_full_tail_matched=baseline_match_count if a.candidate=='A' else None,feedback_steps=feedback,intervention_steps=1,
            exact_origin=True,exact_policy_history=True,extended_environment_exact=True,weights_unchanged=True,continuation_mppi_calls=0)
        dump(out/'episode.json',row);print('DONE',a.scene,a.step,a.candidate,round(row['discounted_return'],4),row['success'],flush=True)
    finally:env.close()
if __name__=='__main__':main()
