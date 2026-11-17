"""Small fresh-state native takeovers; original frozen actor/critic/controller only."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reports/mppi_root_cause_20261117'))
from diagnosis_runtime import make,at_origin,snapshot,physical,dump
import numpy as np,torch
from planning.local_mppi import LocalMPPI
from planning.sac_value import state_sha256,policy_context_sha256
from residual_rl.features import extract_path_features
from tools.policy_value_diagnosis import discounted_return
from common.seed import set_seed


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scenario',type=int,required=True);ap.add_argument('--candidate',choices=['baseline','gaussian1'],required=True)
    ap.add_argument('--policy',choices=['lattice_sac','td_actor'],required=True);ap.add_argument('--repeat',type=int,required=True);ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    p=json.loads((R/'protocol.json').read_text());assert a.scenario in p['fresh_scenarios']
    out=R/'fresh'/f'{a.scenario}_{a.candidate}';out.mkdir(parents=True,exist_ok=True)
    _,cfg,env,agent=make(a.scenario);assert not agent.planner.allow_mppi and agent.planner.allow_residual
    from omegaconf import OmegaConf
    # Actual post-env dimensions/runtime config, unlike unresolved launch YAML.
    if not (out/'runtime_config.json').exists():(out/'runtime_config.json').write_text(json.dumps(vars(cfg) if hasattr(cfg,'__dict__') else dict(cfg),indent=2,default=str))
    def wm_hash():
        # TensorDict serializes non-tensor shape metadata alongside actualweights.
        return state_sha256({n:t for n,t in agent.model.state_dict().items() if torch.is_tensor(t)})
    hashes=dict(wm=wm_hash(),sac={n:state_sha256(getattr(agent.residual_sac,n).state_dict()) for n in ['actor','q1','q2','target_q1','target_q2']})
    runs=[(a.policy,a.repeat,a.replay)]
    reference=json.loads((out/'prefix_origin.json').read_text()) if (out/'prefix_origin.json').exists() else None
    pool_reference=torch.tensor(reference['candidate_actions'],device=agent.device) if reference is not None else None
    terminal_reference=json.loads((out/'terminal_origin.json').read_text()) if (out/'terminal_origin.json').exists() else None
    try:
        for method,rep,replay in runs:
            label=f'{method}_r{rep}'+('_replay' if replay else '')
            assert not (out/(label+'.npz')).exists(), 'Retain existing native trajectory'
            obs,ctx,baseline=at_origin(env,agent,a.scenario,p['before'])
            sampler=LocalMPPI(agent.planner.evaluator,dict(cfg.mpr_mpc['mppi']),horizon=10,action_dim=2,device=agent.device);sampler.reset(1+a.scenario)
            pool=sampler._candidate_batch(baseline,sampler.initial_std_vector.expand_as(baseline),baseline)[:2]
            z0=ctx.latent.clone();commands=pool[p['candidates'].index(a.candidate)]
            start=snapshot(env,agent,obs,pool,a.scenario,p['before'])
            if reference is None:reference=start;pool_reference=pool.clone();dump(out/'prefix_origin.json',start)
            else:assert start==reference and torch.equal(pool,pool_reference),'prefix replay origin mismatch'
            prefix_observations=[obs.cpu().numpy().copy()];prefix_rewards=[];prefix_states=[physical(env)];done=False
            predicted=z0.clone()
            for t in range(10):
                predicted=agent.model.next(predicted,commands[t:t+1],None)
                obs,_,done,info=env.step(commands[t]);prefix_observations.append(obs.cpu().numpy().copy());prefix_rewards.append(float(info['tdmpc2_reward']));prefix_states.append(physical(env))
                if done:break
            if done:
                dump(out/(label+'_unavailable.json'),dict(scenario=a.scenario,candidate=a.candidate,method=method,repeat=rep,prefix_steps=len(prefix_rewards),reason='native termination before nonterminalH state',success=bool(info['success']),out_of_road=bool(info.get('out_of_road',False))))
                continue
            # Prepare the SAME originalLattice+SAC action/state for both readouts and both takeovers.
            sac_first=agent.act(obs,t0=False,eval_mode=True);ctx=agent.planner.last_context;state=ctx.residual_state.clone();assert state.shape==(535,)
            actual_z=ctx.latent.clone();terminal=snapshot(env,agent,obs,commands[None],a.scenario,p['before']+10)
            if terminal_reference is None:
                terminal_reference=terminal;dump(out/'terminal_origin.json',terminal)
                actor=agent.residual_sac.actor(state[None],deterministic=True)[0]
                assert torch.equal(actor.squeeze(0),agent.last_plan.requested_normalized_residual.reshape(-1))
                np.savez_compressed(out/'terminal.npz',predicted_latent=predicted.cpu().numpy().reshape(-1),actual_observation=obs.cpu().numpy(),
                    actual_latent=actual_z.cpu().numpy().reshape(-1),state535=state.cpu().numpy(),actor_output=actor.cpu().numpy().reshape(-1),
                    lattice_features=state[512:524].cpu().numpy(),coarse_actions=ctx.coarse_actions.cpu().numpy(),
                    sac_physical_action=sac_first.cpu().numpy(),commands=commands.cpu().numpy(),prefix_observations=prefix_observations,prefix_rewards=prefix_rewards,
                    prefix_physical=prefix_states,residual_bounds=ctx.residual_bounds.cpu().numpy())
                dump(out/'lattice_features.json',extract_path_features(ctx.selection.coarse_path))
                dump(out/'terminal_metadata.json',dict(scenario_id=a.scenario,origin_id=f'{a.scenario}_30',candidate_id=a.candidate,terminal_step=10,
                    episode_seed=a.scenario,initial_policy_seed=1+a.scenario,policy_context_sha256=policy_context_sha256(cfg),
                    residual_valid=bool(agent.last_plan.residual_valid),actual_lattice_sac_action=sac_first.cpu().tolist()))
            else:
                assert terminal==terminal_reference,'terminal observed/state/traffic/RNG mismatch'
                saved=np.load(out/'terminal.npz');assert np.array_equal(saved['state535'],state.cpu().numpy()) and np.array_equal(saved['sac_physical_action'],sac_first.cpu().numpy())
            # Paired suffix RNG intervention; native traffic internal RNG is unchanged.
            suffix_seed=a.scenario*10+rep;set_seed(suffix_seed)
            dump(out/(label+'_start.json'),snapshot(env,agent,obs,commands[None],a.scenario,p['before']+10))
            observations=[obs.cpu().numpy().copy()];actions=[];rewards=[];costs=[];dones=[];invalid=[];physical_states=[physical(env)];crash=offroad=False
            first_td_action=None
            while not done:
                t=len(actions)
                if method=='lattice_sac':
                    action=sac_first if t==0 else agent.act(obs,t0=False,eval_mode=True)
                    assert agent.last_plan.stage.use_residual and not agent.last_plan.stage.use_mppi
                    bad=not bool(agent.last_plan.residual_valid)
                else:
                    z=agent.planner.evaluator.encode(obs)
                    action=agent.model.pi(z,None)[0][0].cpu()  # ORIGINAL stochastic TD actor
                    bad=False
                    if t==0:first_td_action=action.numpy().copy()
                obs,reward,done,info=env.step(action)
                assert abs(float(reward)-float(info['tdmpc2_reward']))<1e-4
                observations.append(obs.cpu().numpy().copy());actions.append(action.cpu().numpy().copy());rewards.append(float(info['tdmpc2_reward']));costs.append(float(info['cost']));dones.append(bool(done));invalid.append(bad);physical_states.append(physical(env))
                crash|=any(bool(info.get(k,False)) for k in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk']);offroad|=bool(info.get('out_of_road',False))
                assert len(actions)<=1000
                if t and t%150==0:print(a.scenario,a.candidate,label,'suffixstep',t,'route',round(info['route_completion'],3),flush=True)
            np.savez_compressed(out/(label+'.npz'),observations=observations,actions=actions,rewards=rewards,costs=costs,done=dones,invalid_residual=invalid,physical=physical_states)
            row=dict(scenario=a.scenario,candidate=a.candidate,origin_id=f'{a.scenario}_30',policy=method,repeat=rep,suffix_seed=suffix_seed,replay=replay,
                discounted_return=discounted_return(rewards,.995,dones),discounted_return_sac_gamma=discounted_return(rewards,.99,dones),
                episode_return=sum(rewards),success=float(info['success']),crash=float(crash),out_of_road=float(offroad),route_completion=float(info['route_completion']),
                safety_cost=sum(costs),suffix_steps=len(actions),native_done=True,end_reason=info.get('end_reason',''),invalid_residual_count=sum(invalid),
                reward_definition='info.tdmpc2_reward',no_mppi=True,td_first_action=first_td_action.tolist() if first_td_action is not None else None)
            dump(out/(label+'.json'),row);print('DONE',a.scenario,a.candidate,label,'return',round(row['discounted_return'],3),'success',row['success'],flush=True)
        assert hashes['wm']==wm_hash()
        assert all(h==state_sha256(getattr(agent.residual_sac,n).state_dict()) for n,h in hashes['sac'].items())
        dump(out/(label+'_verification.json'),dict(original_hashes=hashes,weights_unchanged=True,prefix_origin_exactly_replayed=True,terminal_state_observation_traffic_RNG_paired=terminal_reference is not None,training=False,mppi=False))
    finally:env.close()

if __name__=='__main__':main()
