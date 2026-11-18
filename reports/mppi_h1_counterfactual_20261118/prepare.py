"""Compute actual C/D outputs at a reconstructed B origin; never execute them here."""
import argparse,time
from study import *

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True);ap.add_argument('--step',type=int,required=True);a=ap.parse_args()
    p=load(R/'protocol.json');assert a.scene in p['scenes'] and a.step in p['steps']
    out=origin_dir(a.scene,a.step);out.mkdir(parents=True,exist_ok=False)
    parsed,cfg,env,agent=make_runtime(a.scene)
    try:
        obs,baseline,trace,state,policy,pre_rng=restore_origin(env,agent,a.scene,a.step)
        post_rng=get_rng();env_state=extended_environment(env)
        dump(out/'origin.json',state);dump(out/'policy.json',policy);dump(out/'environment.json',env_state)
        baseline_sequence=agent.last_plan.refined_actions.clone();state535=agent.planner.last_context.residual_state.clone()
        latent=agent.planner.last_context.latent.clone();private=agent.planner.local_mppi.generator.get_state().clone()
        results={};arrays=dict(baseline_sequence=baseline_sequence.cpu().numpy(),latent=latent.cpu().numpy(),state535=state535.cpu().numpy())
        evaluator=agent.planner.evaluator;original=evaluator.evaluate_mppi_candidates
        for label,cartesian in [('C',False),('D',True)]:
            calls=[]
            def capture(*args,**kwargs):
                result=original(*args,**kwargs);calls.append((args[1].detach().clone(),result));return result
            evaluator.evaluate_mppi_candidates=capture
            agent.planner.allow_mppi=True;agent.planner.mppi_cfg['cartesian_virtual_speed']=cartesian
            agent.planner.local_mppi.generator.set_state(private);set_rng(pre_rng)
            begin=time.perf_counter();selected=agent.act(obs,t0=False,eval_mode=True);torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-begin)
            plan=agent.last_plan;ctx=agent.planner.last_context
            assert plan.stage.use_mppi and torch.equal(ctx.residual_state,state535)
            assert torch.equal(plan.refined_actions[0],baseline_sequence[0])
            assert len(calls)==5 and all(x[0].shape==(65,11,2) for x in calls[:4]) and calls[-1][0].shape==(2,11,2)
            assert torch.equal(selected,plan.debug['selected_mppi_actions'][0])
            last_actions,last=calls[3];final_actions,final=calls[4]
            assert all(x[1].q_member_indices is not None for x in calls)
            arrays.update({f'{label}_iteration_actions':torch.stack([x[0] for x in calls[:4]]).cpu().numpy(),
                f'{label}_iteration_values':torch.stack([x[1].total_value for x in calls[:4]]).cpu().numpy(),
                f'{label}_final_actions':final_actions.cpu().numpy(),f'{label}_final_values':final.total_value.cpu().numpy(),
                f'{label}_selected':plan.debug['selected_mppi_actions'].cpu().numpy()})
            results[label]=dict(action=selected.cpu().numpy(),selected_source=int(plan.metrics['mppi/selected_source_code']),
                score=float(plan.metrics['mppi/selected_wm_value']) if 'mppi/selected_wm_value' in plan.metrics else float(plan.metrics['mpr/selected_wm_value']),
                final_pair=list(final.q_member_indices),iteration_pairs=[list(x[1].q_member_indices) for x in calls[:4]],
                planning_ms=elapsed,baseline_same_first=True,coarse535_bit_exact=True)
            set_rng(post_rng)
            assert snapshot(env,obs)['hashes']==state['hashes'],'Candidate generation mutated branch RNG/physical state'
            assert extended_environment(env)['hash']==env_state['hash'],'Candidate generation mutated simulator/traffic policy history'
        evaluator.evaluate_mppi_candidates=original
        C=arrays['C_selected'];D=arrays['D_selected'];last=arrays['C_iteration_actions'][-1];scores=arrays['C_iteration_values'][-1]
        order=np.argsort(scores[1:],kind='stable')+1
        raw=[dict(role='A0',source='SAC baseline',action=baseline.cpu().numpy(),sequence=baseline_sequence.cpu().numpy(),
                 raw_score=float(arrays['C_final_values'][0]),q_pair=results['C']['final_pair'],candidate_rank=None),
            dict(role='A1',source='C actual selected output',action=C[0],sequence=C,raw_score=results['C']['score'],q_pair=results['C']['final_pair'],candidate_rank=None),
            dict(role='A2',source='D actual selected output',action=D[0],sequence=D,raw_score=results['D']['score'],q_pair=results['D']['final_pair'],candidate_rank=None)]
        for role,q in zip(['A3','A4','A5','A6','A7'],p['gaussian_quantiles']):
            pos=int(round(q*63));idx=int(order[pos])
            raw.append(dict(role=role,source=f'C iteration3 actual Gaussian q{q}',action=last[idx,0],sequence=last[idx],
                raw_score=float(scores[idx]),q_pair=results['C']['iteration_pairs'][3],candidate_rank=64-pos,original_row=idx))
        kept,aliases=representative_roles(raw,p['dedup_l2'])
        sequences=torch.tensor(np.stack([x['sequence'] for x in raw]),device=agent.device)
        same=evaluator.evaluate_mppi_candidates(latent,sequences,compute_cost=False,q_aggregation='fixed_pair',q_indices=tuple(results['C']['final_pair']))
        for i,x in enumerate(raw):x.update(common_h10_score=float(same.total_value[i]),predicted_first_reward=float(same.reward_sequence[i,0]))
        # Populate scores in representatives after the outcome-free common-Q diagnostic pass.
        for x in kept:
            original_row=next(y for y in raw if y['role']==x['candidate_id'])
            x.update(common_h10_score=original_row['common_h10_score'],predicted_first_reward=original_row['predicted_first_reward'])
        set_rng(post_rng);assert snapshot(env,obs)['hashes']==state['hashes']
        arrays['candidate_actions']=np.array([x['action'] for x in kept],np.float32)
        np.savez_compressed(out/'catalog.npz',**arrays)
        dump(out/'catalog.json',dict(scene=a.scene,step=a.step,representatives=[{k:v for k,v in x.items() if k!='sequence'} for x in kept],
            raw_roles=[{k:v for k,v in x.items() if k!='sequence'} for x in raw],aliases=aliases,
            C=results['C'],D=results['D'],formal_H1_ranking='NOT ESTIMABLE',candidate_generation_env_rng_unchanged=True,
            candidate_executor_process_isolation=True,origin_prefix_verified=True,policy_fingerprint=policy['hashes']))
        print('PREPARED',a.scene,a.step,'unique',len(kept),'roles',aliases,flush=True)
    finally:env.close()
if __name__=='__main__':main()
