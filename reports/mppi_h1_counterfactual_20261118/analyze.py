"""Map-cluster paired H1 effects. H10/H1 correlations are explicitly descriptive, never ranking validation."""
from collections import defaultdict
import pandas as pd
from scipy.stats import spearmanr
from study import *

def bootstrap_map(frame,metrics):
    maps=frame.groupby('scene')[metrics].mean();rng=np.random.default_rng(20261118)
    indices=rng.integers(len(maps),size=(2000,len(maps)))
    answer={}
    for name in metrics:
        v=maps[name].to_numpy(float);samples=v[indices].mean(1)
        answer[name]=dict(mean=float(v.mean()),ci95=np.quantile(samples,[.025,.975]),
            per_map={int(s):float(x) for s,x in maps[name].items()},
            leave_one_map_out={int(s):float(np.delete(v,i).mean()) for i,s in enumerate(maps.index)})
    return answer

def main():
    complete=load(R/'collection_complete.json');assert complete['origins']==16
    protocol=load(R/'protocol.json');actions=[];episodes=[];stepframes=[];origins=[];roles=[];ranking=[];reconstruction=[]
    for scene in protocol['scenes']:
        for step in protocol['steps']:
            directory=origin_dir(scene,step);meta=load(directory/'catalog.json');basepath=R/'branches'/f'{scene}_{step}'
            records={}
            a0=load(basepath/'A0'/'episode.json');base=a0['discounted_return']
            for candidate in meta['representatives']:
                cid=candidate['candidate_id'];out=basepath/cid;row=load(out/'episode.json')
                assert row['exact_origin'] and row['exact_policy_history'] and row['extended_environment_exact'] and row['native_done'] and row['weights_unchanged']
                assert row['intervention_steps']==1 and row['continuation_mppi_calls']==0 and row['feedback_steps']==row['episode_length']-1
                z=np.load(out/'trace.npz');assert discounted_return(z['rewards'])==row['discounted_return']
                assert np.array_equal(z['actions'][0],np.asarray(candidate['action'],np.float32)) and not z['done'][:-1].any() and z['done'][-1]
                audit=load(out/'origin_verification.json');assert audit['physical_rng_hashes']==load(directory/'origin.json')['hashes']
                assert audit['policy_hashes']==meta['policy_fingerprint'] and audit['extended_environment_hash']==load(directory/'environment.json')['hash']
                records[cid]=row
                immediate=row.pop('immediate');record={k:(json.dumps(clean(v)) if isinstance(v,(list,dict)) else v) for k,v in row.items()}
                record.update({f'immediate_{k}':v for k,v in immediate.items()});record['actual_gain']=row['discounted_return']-base
                record.update(candidate_source=candidate['source'],predicted_common_h10_score=candidate['common_h10_score'],
                    predicted_first_reward=candidate['predicted_first_reward'],immediate_reward_error=candidate['predicted_first_reward']-immediate['reward'],
                    formal_h1_ranking='NOT ESTIMABLE')
                episodes.append(record)
                f=pd.read_csv(out/'steps.csv');assert f.intervention.sum()==1 and f.sac_takeover.sum()==len(f)-1
                assert len(f)==row['episode_length'];stepframes.append(f)
                actions.append(dict(scene=scene,step=step,candidate_id=cid,candidate_source=candidate['source'],
                    aliases=json.dumps(candidate['roles']),normalized_steering=candidate['action'][0],normalized_throttle_brake=candidate['action'][1],
                    predicted_score=candidate['raw_score'],q_pair=json.dumps(candidate['q_pair']),candidate_rank=candidate['candidate_rank'],
                    common_h10_score=candidate['common_h10_score'],predicted_first_reward=candidate['predicted_first_reward'],
                    action_distance_to_baseline=float(np.linalg.norm(np.asarray(candidate['action'])-np.asarray(meta['representatives'][0]['action']))),
                    submitted_action_equal=True,score_target='H10 prefix + TD-Q, not H1 + SAC'))
            ys=np.array([records[x['candidate_id']]['discounted_return'] for x in meta['representatives']]);best=float(ys.max());ids=list(records)
            c=records[meta['aliases']['A1']];d=records[meta['aliases']['A2']]
            nonbaseline=[r for cid,r in records.items() if cid!='A0'];gains=[r['discounted_return']-base for r in nonbaseline]
            row=dict(scene=scene,step=step,unique_candidates=len(records),baseline_return=base,
                oracle_candidate=ids[int(ys.argmax())],oracle_return=best,oracle_gain=best-base,
                C_return=c['discounted_return'],D_return=d['discounted_return'],C_selected_gain=c['discounted_return']-base,D_selected_gain=d['discounted_return']-base,
                D_minus_C_gain=d['discounted_return']-c['discounted_return'],
                C_top1_regret=best-c['discounted_return'],D_top1_regret=best-d['discounted_return'],
                random_candidate_mean_gain=float(np.mean(gains)) if gains else 0.,
                positive_candidate_ratio=float(np.mean(np.array(gains)>0)) if gains else 0.,
                robust_positive_candidate_ratio=float(np.mean(np.array(gains)>1)) if gains else 0.,
                oracle_positive=float(best>base),oracle_gain_over_one=float(best-base>1),
                C_selected_positive=float(c['discounted_return']>base),D_selected_positive=float(d['discounted_return']>base),
                C_selected_negative_over_one=float(c['discounted_return']-base < -1),D_selected_negative_over_one=float(d['discounted_return']-base < -1))
            for mode,method in [('B',a0),('C',c),('D',d)]:
                role={k:method[k] for k in ['success','crash','out_of_road','route_completion','timeout','safety_cost','episode_length','episode_return','discounted_return']}
                roles.append(dict(scene=scene,step=step,configuration=mode,representative=meta['aliases'][{'B':'A0','C':'A1','D':'A2'}[mode]],**role))
                if mode!='B':
                    for key in ['success','crash','out_of_road','route_completion','timeout','safety_cost']:
                        row[f'{mode}_{key}_delta']=method[key]-a0[key]
            predictions=np.array([x['common_h10_score'] for x in meta['representatives']])
            descriptive=float(spearmanr(predictions,ys).statistic) if np.ptp(ys)>0 and np.ptp(predictions)>0 else None
            row['descriptive_H10_score_vs_H1_return_spearman']=descriptive
            origins.append(row)
            for candidate,y in zip(meta['representatives'],ys):
                ranking.append(dict(scene=scene,step=step,candidate_id=candidate['candidate_id'],actual_return=y,
                    actual_gain=y-base,predicted_common_h10_score=candidate['common_h10_score'],
                    actual_finite_pool_rank=int(1+np.sum(ys>y)),predicted_h10_descriptive_rank=int(1+np.sum(predictions>candidate['common_h10_score'])),
                    formal_H1_spearman='NOT ESTIMABLE',formal_H1_kendall='NOT ESTIMABLE',formal_H1_pair_accuracy='NOT ESTIMABLE',
                    reason='No validated value for original Lattice+SAC continuation; H10 TD-Q target differs.'))
            reconstruction.append(dict(scene=scene,step=step,prefix_controls=step-1,baseline_tail_exact=a0['baseline_full_tail_matched'],
                exact_observation_vehicle_traffic_rng=True,exact_latent_residual535_policy=True,all_extended_environment_hashes_equal=True,
                unique_executed_branches=len(records),formal_h1_ranking='NOT ESTIMABLE'))
    assert len(episodes)==complete['branches']
    table(R/'candidate_actions.csv',actions);table(R/'counterfactual_episodes.csv',episodes)
    pd.concat(stepframes,ignore_index=True).to_csv(R/'counterfactual_steps.csv',index=False,mode='x')
    table(R/'oracle_gain_analysis.csv',origins);table(R/'candidate_ranking.csv',ranking);table(R/'role_results.csv',roles)
    table(R/'state_reconstruction_checks.csv',reconstruction)
    o=pd.DataFrame(origins);numeric=[k for k in o.columns if k not in ['scene','step','oracle_candidate','descriptive_H10_score_vs_H1_return_spearman']]
    maps=o.groupby('scene')[numeric].mean().reset_index();maps.to_csv(R/'map_level_statistics.csv',index=False,mode='x')
    metric_names=[k for k in numeric if k not in ['unique_candidates','baseline_return','oracle_return','C_return','D_return']]
    stats=bootstrap_map(o,metric_names)
    role_table=pd.DataFrame(roles)
    dump(R/'paired_statistics.json',dict(maps=8,origins=16,branches=len(episodes),bootstrap_samples=2000,seed=20261118,
        unit='8 maps; both origins and all paired candidate outcomes resampled together',metrics=stats,
        strict_positive_origins=int(o.oracle_positive.sum()),gain_over_one_origins=int(o.oracle_gain_over_one.sum()),
        C_positive_origins=int(o.C_selected_positive.sum()),D_positive_origins=int(o.D_selected_positive.sum()),
        C_positive_when_oracle_over_one=int(((o.C_selected_gain>1)&(o.oracle_gain>1)).sum()),
        D_positive_when_oracle_over_one=int(((o.D_selected_gain>1)&(o.oracle_gain>1)).sum()),
        formal_h1_ranking='NOT ESTIMABLE',formal_rank_metrics=None,
        reason='Frozen TD terminal Q and historical unvalidated continuation heads cannot represent the required H1 continuation target.',
        immediate_reward_mae=float(np.mean(np.abs([r['immediate_reward_error'] for r in episodes]))),
        role_means=role_table.groupby('configuration').mean(numeric_only=True).to_dict('index'),
        warning='Known development maps, finite post-hoc oracle, single traffic seed,8clusters; no independent online deployment claim.'))
    print('H1_STATISTICS',len(episodes),flush=True)
    for k in ['oracle_gain','C_selected_gain','D_selected_gain','C_top1_regret','D_top1_regret']:
        print(k,clean(stats[k]),flush=True)

if __name__=='__main__':main()
