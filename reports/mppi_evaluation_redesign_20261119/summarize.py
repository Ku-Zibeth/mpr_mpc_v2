"""Actual108 native closed-loop screening, paired by map+seed. No inferred counterfactual labels online."""
import pandas as pd
from h1_analysis import clustered
from study import *

def main():
    assert (R/'probe_complete.json').exists();p=load(R/'protocol.json');decision=load(R/'decision.json');episodes=[];frames=[];data={}
    keys=['success','crash','out_of_road','route_completion','timeout','episode_return','discounted_return','episode_length','safety_cost','replacement_rate','mean_action_deviation']
    for group in ['A','B','C']:
        for scene in p['scenes']:
            for seed in p['seeds']:
                out=original_dir(scene,seed) if group=='A' else R/'closed'/f'{group}_{scene}_r{seed}'
                row=load(out/'episode.json');assert row['native_done'] and row['weights_unchanged']
                data[group,scene,seed]=row
                episodes.append({k:v for k,v in row.items() if k!='initial_hashes'})
                f=pd.read_csv(out/'steps.csv')
                if 'local_mppi_ms' not in f:f['local_mppi_ms']=0. # A has zero MPPI invocations by construction.
                frames.append(f)
                trace=np.load(out/'trace.npz');assert discounted_return(trace['rewards'])==row['discounted_return']
    for scene in p['scenes']:
        for seed in p['seeds']:
            assert data['A',scene,seed]['initial_hashes']==data['B',scene,seed]['initial_hashes']==data['C',scene,seed]['initial_hashes']
            if not decision['online_options']['replacement_supported']:
                a=np.load(original_dir(scene,seed)/'trace.npz');c=np.load(R/'closed'/f'C_{scene}_r{seed}'/'trace.npz')
                assert np.array_equal(a['actions'],c['actions']) and np.array_equal(a['observations'],c['observations']), 'Unsupported guard must preserve exact original SAC trajectory'
    table(R/'paired_closed_loop_episodes.csv',episodes)
    e=pd.DataFrame(episodes);steps=pd.concat(frames,ignore_index=True);steps.to_csv(R/'closed_loop_steps.csv',index=False,mode='x')
    summary=[];statistics={}
    for group in ['A','B','C']:
        rows=e[e.configuration==group];s=steps[steps.configuration==group]
        summary.append(dict(experiment='full_native_closed_loop',configuration=group,comparison='absolute mean',maps=12,cases=36,
            **{k:float(rows[k].mean()) for k in keys},planner_p50_ms=float(s.planner_ms.quantile(.5)),planner_p95_ms=float(s.planner_ms.quantile(.95)),
            control_cycle_p95_ms=float(s.control_cycle_ms.quantile(.95)),timing_scope='6worker shared load, includes cold start'))
    for left,right in [('B','A'),('C','A'),('C','B')]:
        l=e[e.configuration==left].set_index(['scene','seed']);r=e[e.configuration==right].set_index(['scene','seed']);diff=(l[keys]-r[keys]).reset_index()
        stats=clustered(diff,keys);stats['per_seed_gain']=diff.groupby('seed').discounted_return.mean().to_dict();statistics[left+'-'+right]=stats
        summary.append(dict(experiment='full_native_closed_loop',configuration=left,comparison=f'paired difference vs {right}',maps=12,cases=36,
            **{k:v['mean'] for k,v in stats.items() if k in keys}))
    table(R/'summary.csv',summary);dump(R/'closed_statistics.json',dict(comparisons=statistics,maps=12,seeds=3,episodes=108,unit='12 map clusters, all3paired seeds'))
    timing=[]
    probes=pd.concat([pd.read_csv(f) for f in R.glob('probe_*.csv')],ignore_index=True)
    probes.to_csv(R/'isolated_planner_samples.csv',index=False,mode='x')
    for scope,frame in [('shared_native_all',steps),('isolated_frozen_state',probes)]:
        for group,rows in frame.groupby('configuration'):
            for key in [x for x in rows.columns if x.endswith('_ms')]:
                v=pd.to_numeric(rows[key],errors='coerce').dropna()
                if len(v):timing.append(dict(configuration=group,scope=scope,stage=key,samples=len(v),mean_ms=float(v.mean()),p50_ms=float(v.quantile(.5)),p95_ms=float(v.quantile(.95)),p99_ms=float(v.quantile(.99)),maximum_ms=float(v.max())))
    table(R/'latency.csv',timing)
    isolated={g:float(probes[probes.configuration==g].planner_ms.quantile(.95)) for g in ['A','B','C']}
    ca=statistics['C-A'];replacement=float(e[e.configuration=='C'].replacement_rate.mean())
    h1=load(R/'h1_statistics.json');role='D' if decision['branch']=='reward_only' else 'F' if decision['branch']=='validated_margin' else 'A'
    hr=pd.read_csv(R/'h1_role_results.csv');mod=hr[hr.method==role].set_index(['scene','seed']);rnd=hr[hr.method=='C'].set_index(['scene','seed'])
    hd=(mod[['gain']]-rnd[['gain']]).reset_index();h1diff=clustered(hd,['gain'])['gain']
    conditions=dict(success_positive=ca['success']['mean']>0,crash_not_higher=ca['crash']['mean']<=0,oor_not_higher=ca['out_of_road']['mean']<=0,
        discounted_gain_positive=ca['discounted_return']['mean']>0,discounted_gain_ci_positive=ca['discounted_return']['ci95'][0]>0,
        route_not_lower=ca['route_completion']['mean']>=0,not_one_map_only=ca['discounted_return']['positive_maps']>=8,
        h1_beats_random=h1diff['ci95'][0]>0,nonzero_physical_replacement=replacement>0,
        isolated_full_planner_p95_below100=isolated['C']<=100)
    dump(R/'acceptance.json',dict(conditions=conditions,accepted_for_larger_experiment=all(conditions.values()),
        modified_h1_role=role,modified_h1_minus_random=h1diff,replacement_rate=replacement,isolated_planner_p95_ms=isolated,
        protection_not_performance_success=not decision['online_options']['replacement_supported'],
        full_cycle_timing_note='Native cycle=full agent.act + env.step under shared load; isolated probe is planner-only, not full-cycle certificate.'))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':main()
