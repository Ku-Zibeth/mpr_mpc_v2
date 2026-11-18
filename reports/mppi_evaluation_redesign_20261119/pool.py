"""Single unchanged native search at the common B state, then outcome-free same-pool selectors."""
import argparse
from study import *

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    out=origin_dir(a.scene,a.seed);out.mkdir(parents=True,exist_ok=False)
    base=load(original_dir(a.scene,a.seed)/'episode.json')
    if not base['reachable_h1_origin']:
        dump(out/'unreachable.json',dict(scene=a.scene,seed=a.seed,reason='Original baseline terminated before preregistered50',length=base['episode_length']));return
    parsed,cfg,env,agent=make_runtime(a.scene,a.seed)
    try:
        obs,baseline,trace,state,policy,pre_rng=restore_origin(env,agent,a.scene,a.seed);post_rng=get_rng();extended=extended_environment(env)
        assert policy['hashes']==load(original_dir(a.scene,a.seed)/'policy.json')['hashes']
        assert extended['hash']==load(original_dir(a.scene,a.seed)/'environment.json')['hash']
        dump(out/'origin.json',state);dump(out/'policy.json',policy);dump(out/'environment.json',extended)
        calls=[];original=agent.planner.evaluator.evaluate_mppi_candidates
        def capture(*args,**kwargs):
            result=original(*args,**kwargs);calls.append((args[1].clone(),result));return result
        agent.planner.evaluator.evaluate_mppi_candidates=capture;agent.planner.allow_mppi=True;set_rng(pre_rng)
        selected=agent.act(obs,t0=False,eval_mode=True);plan=agent.last_plan
        assert len(calls)==5 and all(x[0].shape==(65,11,2) for x in calls[:4]) and calls[-1][0].shape==(2,11,2)
        final_actions,final=calls[-1];native=0 if bool(final.total_value[0]>=final.total_value[1]) else 1
        assert torch.equal(selected.cpu(),final_actions[native,0].cpu()),'B/E native final selection must reproduce'
        assert torch.equal(final_actions[0,0].cpu(),baseline.cpu())
        evaluator=agent.planner.evaluator;evaluator.evaluate_mppi_candidates=original
        cached=[(final,slice(None)),*[(x[1],slice(1,None)) for x in calls[:4]]]
        pool=torch.cat([final_actions,*[x[0][1:] for x in calls[:4]]])
        same=evaluator.rescore_cached_candidates(cached,q_aggregation='fixed_pair',q_indices=tuple(final.q_member_indices),cost_weight=0.)
        assert pool.shape==(258,11,2)
        controls=pool[:,0].cpu().numpy();unique=[]
        for i in range(1,len(pool)):
            if np.linalg.norm(controls[i]-controls[0])<=1e-5:continue
            if not any(np.linalg.norm(controls[i]-controls[j])<=1e-5 for j in unique):unique.append(i)
        assert unique
        random_id=int(np.random.default_rng(20261119+a.scene*31+a.seed).choice(unique))
        reward_id=int(same.discounted_reward.argmax());rq_id=int(same.total_value.argmax())
        margin=load(R/'frozen_margin.json')['margin'];gain=float(final.total_value[1]-final.total_value[0])
        selectors=dict(A=0,B=native,C=random_id,D=reward_id,E=native,F=1 if gain>margin else 0,G=rq_id)
        raw=[dict(role=role,source={'A':'original SAC','B':'native actual','C':'uniform unique nonBaseline','D':'same-pool Reward-only','E':'native RQ final2 reproduction','F':'frozen margin native mean','G':'supplement matched-pool RQ'}[role],
            action=controls[i],pool_index=i,raw_score=float(same.total_value[i]),candidate_rank=None,
            reward_score=float(same.discounted_reward[i]),terminal_contribution=float(same.total_value[i]-same.discounted_reward[i])) for role,i in selectors.items()]
        representatives,aliases=representative_roles(raw,1e-5)
        set_rng(post_rng);assert snapshot(env,obs)['hashes']==state['hashes'] and extended_environment(env)['hash']==extended['hash']
        np.savez_compressed(out/'pool.npz',pool=pool.cpu().numpy(),scores=same.total_value.cpu().numpy(),reward=same.discounted_reward.cpu().numpy(),
            terminal_q=same.terminal_q.cpu().numpy(),terminal_discount=same.terminal_discount.cpu().numpy(),terminal_latents=same.terminal_latent.cpu().numpy(),
            iteration_actions=torch.stack([x[0] for x in calls[:4]]).cpu().numpy(),iteration_values=torch.stack([x[1].total_value for x in calls[:4]]).cpu().numpy(),
            final_values=final.total_value.cpu().numpy(),latent=agent.planner.last_context.latent.cpu().numpy())
        dump(out/'catalog.json',dict(scene=a.scene,seed=a.seed,step=50,representatives=representatives,raw_roles=raw,aliases=aliases,selectors=selectors,
            native_E_verified=True,final_q_pair=list(final.q_member_indices),iteration_q_pairs=[list(x[1].q_member_indices) for x in calls[:4]],
            pool_size=258,unique_nonbaseline_controls=len(unique),policy_fingerprint=policy['hashes'],candidate_generation_rng_unchanged=True,
            frozen_margin=margin,native_predicted_margin=gain,
            score_semantics='H10 + TD-Q and reward-only are descriptive selectors for H1 labels; no matched learned H1 value is asserted'))
        print('POOL_DONE',a.scene,a.seed,'unique method actions',len(representatives),selectors,flush=True)
    finally:env.close()
if __name__=='__main__':main()
