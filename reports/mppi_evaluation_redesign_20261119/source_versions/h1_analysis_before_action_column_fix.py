"""Paired new-map H1 selectors; selects only the predeclared mechanism branch, never retunes margin."""
import pandas as pd
from study import *

def clustered(frame,columns):
    maps=frame.groupby('scene')[columns].mean();rng=np.random.default_rng(20261119)
    indices=rng.integers(len(maps),size=(2000,len(maps)));result={}
    for key in columns:
        v=maps[key].to_numpy(float)
        result[key]=dict(mean=float(v.mean()),ci95=np.quantile(v[indices].mean(1),[.025,.975]),
            maps={int(s):float(x) for s,x in maps[key].items()},positive_maps=int((v>0).sum()),
            lomo={int(s):float(np.delete(v,i).mean()) for i,s in enumerate(maps.index)})
    return result

def main():
    done=load(R/'h1_complete.json');p=load(R/'protocol.json');results=[];comparisons=[];rqrows=[];coverage=[];roles=[]
    metrics=['discounted_return','success','crash','out_of_road','route_completion','safety_cost','timeout','episode_length','episode_return']
    for scene in p['scenes']:
        for seed in p['seeds']:
            directory=origin_dir(scene,seed)
            if (directory/'unreachable.json').exists():continue
            meta=load(directory/'catalog.json');z=np.load(directory/'pool.npz');byid={}
            assert meta['native_E_verified'] and meta['aliases']['B']==meta['aliases']['E']
            for candidate in meta['representatives']:
                cid=candidate['candidate_id'];out=R/'branches'/f'{scene}_r{seed}'/cid;row=load(out/'episode.json');origin=load(out/'origin_verification.json')
                assert row['exact_origin'] and row['exact_policy_history'] and row['extended_environment_exact'] and row['continuation_mppi_calls']==0 and row['intervention_steps']==1
                assert origin['physical_rng_hashes']==load(directory/'origin.json')['hashes'] and origin['policy_hashes']==meta['policy_fingerprint']
                raw=np.load(out/'trace.npz');assert discounted_return(raw['rewards'])==row['discounted_return']
                byid[cid]=row
            base=byid[meta['aliases']['A']];best=max(x['discounted_return'] for x in byid.values())
            for cid,row in byid.items():
                results.append(dict(scene=scene,seed=seed,candidate_id=cid,roles=json.dumps(row['roles']),**{k:row[k] for k in metrics},
                    steering=candidate['action'][0],throttle_brake=candidate['action'][1],
                    first_action_l2=float(np.linalg.norm(np.asarray(candidate['action'])-np.asarray(meta['raw_roles'][0]['action']))),
                    actual_h1_gain=row['discounted_return']-base['discounted_return'],intervention_steps=1,continuation='original frozen SAC',
                    oracle_scope='max over measured METHOD first-actions only, not all258 unexecuted candidates'))
            rc={role:byid[cid] for role,cid in meta['aliases'].items()}
            for role,row in rc.items():
                rawrole=next(x for x in meta['raw_roles'] if x['role']==role)
                action=np.asarray(rawrole['action']);baseline_action=np.asarray(meta['raw_roles'][0]['action'])
                roles.append(dict(scene=scene,seed=seed,method=role,representative=meta['aliases'][role],
                    gain=row['discounted_return']-base['discounted_return'],regret=best-row['discounted_return'],
                    physical_replacement=float(np.linalg.norm(action-baseline_action)>1e-5),negative_selected=float(row['discounted_return']-base['discounted_return']<-1),
                    **{k:row[k] for k in metrics},oracle_gain=best-base['discounted_return']))
            def difference(left,right):return {k:rc[left][k]-rc[right][k] for k in metrics}
            c=dict(scene=scene,seed=seed,**difference('B','C'));comparisons.append(c)
            rqrows.append(dict(scene=scene,seed=seed,comparison='D_reward_vs_G_same_pool_RQ',**difference('D','G')))
            rqrows.append(dict(scene=scene,seed=seed,comparison='D_reward_vs_B_native_final2',**difference('D','B')))
            for iteration,batch in enumerate(z['iteration_actions']):
                controls=batch[1:,0];delta=controls-batch[0,0]
                coverage.append(dict(scene=scene,seed=seed,iteration=iteration,unique_first_ratio=len(np.unique(controls,axis=0))/64,
                    first_l2_mean=float(np.linalg.norm(delta,axis=1).mean()),steering_std=float(controls[:,0].std()),throttle_std=float(controls[:,1].std()),
                    pool_size=258,measured_method_actions=len(byid),measured_positive_ratio=float(np.mean([x['discounted_return']>base['discounted_return'] for cid,x in byid.items() if cid!='A'])),
                    full_pool_actual_positive_ratio=None,unobserved_actual_candidates='Do not invent actual labels for the unexecuted pool'))
    table(R/'new_map_candidate_results.csv',results);table(R/'h1_role_results.csv',roles)
    table(R/'random_vs_native.csv',comparisons);table(R/'reward_only_vs_reward_q.csv',rqrows);table(R/'candidate_coverage.csv',coverage)
    frame=pd.DataFrame(roles);statistics={};paired={}
    for role,group in frame.groupby('method'):
        statistics[role]=clustered(group,['gain','regret','physical_replacement','negative_selected','oracle_gain',*metrics])
    for left,right in [('B','C'),('D','G'),('D','B'),('F','A'),('G','C')]:
        l=frame[frame.method==left].set_index(['scene','seed']);r=frame[frame.method==right].set_index(['scene','seed'])
        diff=(l[['gain','regret','crash','out_of_road','success']]-r[['gain','regret','crash','out_of_road','success']]).reset_index()
        paired[left+'-'+right]=clustered(diff,['gain','regret','crash','out_of_road','success'])
    dump(R/'h1_statistics.json',dict(maps=int(frame.scene.nunique()),cases=int(len(frame[frame.method=='A'])),branches=done['branches'],
        selectors=statistics,paired=paired,bootstrap_unit='complete map, all3seeds together',bootstrap_samples=2000,
        formal_h1_model_ranking='NOT ESTIMABLE; selector consequences are directly measured, H10 score correlations remain descriptive',
        finite_oracle='Measured METHOD actions, not whole258pool'))
    # Retain development table verbatim; append ONE frozen-margin independent screen.
    dev=R/'baseline_margin_analysis.csv';dev.rename(R/'baseline_margin_development.csv')
    old=pd.read_csv(R/'baseline_margin_development.csv').to_dict('records');f=statistics['F']
    old.append(dict(margin=p['margin'],replacement_rate=f['physical_replacement']['mean'],selected_gain=f['gain']['mean'],
        negative_selection_rate=f['negative_selected']['mean'],source='new12maps3seeds ONE frozen margin; no retuning',eligible_nontrivial=f['physical_replacement']['mean']>0))
    table(dev,old)
    reward_ok=(statistics['D']['gain']['ci95'][0]>0 and paired['D-G']['gain']['ci95'][0]>0 and paired['D-B']['gain']['ci95'][0]>0
        and statistics['D']['gain']['positive_maps']>=8 and paired['D-G']['crash']['mean']<=0 and paired['D-G']['out_of_road']['mean']<=0
        and statistics['D']['crash']['mean']<=statistics['A']['crash']['mean'] and statistics['D']['out_of_road']['mean']<=statistics['A']['out_of_road']['mean'])
    margin_ok=(f['gain']['ci95'][0]>0 and f['physical_replacement']['mean']>0 and paired['F-A']['crash']['mean']<=0 and paired['F-A']['out_of_road']['mean']<=0)
    if reward_ok:
        branch='reward_only';options=dict(final_selection_policy='reward_pool',final_selection_margin=0.,replacement_supported=True)
    elif margin_ok:
        branch='validated_margin';options=dict(final_selection_policy='baseline_guard',final_selection_margin=p['margin'],replacement_supported=True)
    else:
        branch='unsupported_replacement_protection';options=dict(final_selection_policy='baseline_guard',final_selection_margin=p['margin'],replacement_supported=False)
    dump(R/'decision.json',dict(branch=branch,online_options=options,reward_case_passed=bool(reward_ok),margin_case_passed=bool(margin_ok),
        margin_frozen_before_new_maps=p['margin'],new_margin_tuning=False,default_config_changed=False,
        protection_is_not_algorithm_success=not(reward_ok or margin_ok),
        rationale='Apply the predeclared branch only; unsupported scoring cannot justify replacing SAC. Full closed-loop screening still required.'))
    print('H1_DECISION',branch,options,flush=True)
    for role in ['B','C','D','F','G']:print(role,statistics[role]['gain'],flush=True)
if __name__=='__main__':main()
