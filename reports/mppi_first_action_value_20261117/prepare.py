"""One actual nativeLocalMPPI search per predeclared origin; save no-future features."""
import argparse,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *
from planning.local_mppi import LocalMPPI

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True);ap.add_argument('--before',type=int,required=True);a=ap.parse_args()
    p=load(REPORT/'protocol.json');assert a.scene in p['pilot_scenes']+p['extra_development_scenes']+p['validation_scenes'];assert a.before in p['before_steps']
    out=REPORT/'origins'/f'{a.scene}_{a.before}';out.mkdir(parents=True,exist_ok=False)
    cfg,env,agent=make_runtime(a.scene)
    try:
        obs,unreachable=at_origin(env,agent,a.scene,a.before)
        if unreachable:dump(out/'unreachable.json',unreachable|dict(scene=a.scene,before=a.before));return
        ctx=agent.planner.last_context;plan=agent.last_plan;baseline=plan.refined_actions.clone();z=ctx.latent.clone();s=ctx.residual_state.clone()
        path=plan.refined_path;params=np.array([path.target_d,path.target_speed,path.horizon],np.float32)
        original=agent.planner.evaluator.evaluate_mppi_candidates;calls=[]
        def capture(*args,**kwargs):
            result=original(*args,**kwargs);calls.append((args[1].clone(),result));return result
        options=dict(cfg.mpr_mpc['mppi']);options.update(record_candidates=True,audit_candidates=False,record_cost=False,cost_weight=0.)
        local=LocalMPPI(agent.planner.evaluator,options,horizon=10,action_dim=2,device=agent.device);local.reset(1+a.scene)
        agent.planner.evaluator.evaluate_mppi_candidates=capture
        torch.cuda.synchronize();begin=time.perf_counter();result=local.plan(z,baseline,eval_mode=True);torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)*1000
        agent.planner.evaluator.evaluate_mppi_candidates=original
        assert len(calls)==5 and len(calls[0][0])==65 and not result.numerical_error
        final=calls[-1][1];pair=final.q_member_indices;assert pair is not None and len(pair)==2
        pool=torch.cat([calls[-1][0],*[seq[1:] for seq,_ in calls[:-1]]])
        common=agent.planner.evaluator.rescore_cached_candidates([(final,slice(None)),*[(r,slice(1,None)) for _,r in calls[:-1]]],q_aggregation='fixed_pair',q_indices=pair)
        selected=0 if result.baseline_selected else 1
        assert torch.equal(result.selected_sequence,pool[selected])
        representatives,aliases=representative_candidates(pool.cpu().numpy(),common.total_value.cpu().numpy(),selected,20261117+a.scene+a.before)
        full=pool.cpu().numpy();ids=[r['pool_index'] for r in representatives];features={k:np.stack([feature_vectors(z.cpu().numpy(),s.cpu().numpy(),params,baseline[0].cpu().numpy(),full[i,0])[k] for i in ids]) for k in ['F1','F2','F3']}
        # All following analyses use cachedarrays/forkedRNG; this is the exactstate branches must reproduce.
        state=snapshot(env,obs);dump(out/'origin.json',state)
        np.savez_compressed(out/'catalog.npz',pool=full,actions=full[ids],pool_indices=ids,z0=z.cpu().numpy().reshape(-1),state535=s.cpu().numpy(),refined_parameters=params,
            original_refined_parameters=plan.refined_parameters,baseline=baseline.cpu().numpy(),selected=result.selected_sequence.cpu().numpy(),
            scores=common.total_value.cpu().numpy(),prefix=common.discounted_reward.cpu().numpy(),discount=common.terminal_discount.cpu().numpy(),
            terminal_latents=common.terminal_latent.cpu().numpy(),q_members=common.q_member_values.cpu().numpy().T,
            raw_final_values=final.total_value.cpu().numpy(),actual_final_pair=pair,
            iteration_actions=np.stack([seq.cpu().numpy() for seq,_ in calls[:-1]]),iteration_values=np.stack([r.total_value.cpu().numpy() for _,r in calls[:-1]]),
            iteration_q_pairs=np.asarray([r.q_member_indices for _,r in calls[:-1]]),post_cpu_rng=torch.random.get_rng_state().numpy(),post_cuda_rng=torch.cuda.get_rng_state().cpu().numpy(),**features)
        dump(out/'catalog.json',dict(scene=a.scene,before=a.before,representatives=representatives,aliases=aliases,unique=len(ids),roles_before_dedup=8,
            pool_size=len(full),selected_pool_index=selected,actual_final_pair=pair,iteration_q_pairs=[r.q_member_indices for _,r in calls[:-1]],
            selection_mode=local.selection_mode,legacy_default_preserved=not local.compare_final_elite,planning_ms=elapsed,
            features_available_before_any_candidate_environment_action=True,feature_dimensions={k:v.shape[1] for k,v in features.items()},
            refined_actual_not_coarse=True,residual_valid=bool(plan.residual_valid),baseline_metrics=plan.metrics,
            native_config={k:getattr(cfg,k) for k in ['mpc','horizon','num_samples','num_elites','iterations','temperature','min_std','max_std']},
            simulator=cfg.metadrive['simulator']))
        print('PREPARED',a.scene,a.before,'unique',len(ids),'local_ms',round(elapsed,2),flush=True)
    finally:env.close()

if __name__=='__main__':main()
