"""Prespecified paired map-cluster comfort/task comparisons; no tuning or control writes."""
import pandas as pd
from comfort_study import *

def main():
    p=load(R/'protocol.json');assert load(R/'closed_complete.json')['episodes']==108
    rows=[];steps=[];signatures={};exactpairs=0
    for scene in p['scenes']:
        for seed in p['seeds']:
            initial=[]
            for group in ['A','B','C']:
                out=R/'episodes'/f'{group}_{scene}_r{seed}';d=load(out/'episode.json');assert d['native_done'] and d['weights_unchanged']
                initial.append(d.pop('initial_hashes'));rows.append(d);steps.append(pd.read_csv(out/'steps.csv'))
                identity=load(out/'map_identity.json');assert identity['current_seed']==scene
                signatures.setdefault(scene,set()).add(identity['road_geometry_sha256'])
                t=np.load(out/'trace.npz');assert discounted(t['rewards'])==d['discounted_return']
                f=pd.read_csv(out/'steps.csv');assert np.array_equal(t['actions'][:,1],f.sac_throttle.to_numpy(dtype=np.float32))
                max_steer=np.deg2rad(load(R/'vehicle_parameters.json')['max_steering'])
                assert np.allclose(f.front_wheel_angle_rad,f.steering*max_steer,rtol=0,atol=1e-6)
            assert initial[0]==initial[1]==initial[2],(scene,seed,'initial pairing');exactpairs+=1
    assert all(len(x)==1 for x in signatures.values()) and len({next(iter(x)) for x in signatures.values()})==12
    frame=pd.DataFrame(rows);frame.to_csv(R/'paired_closed_loop_episodes.csv',index=False,mode='x');step=pd.concat(steps,ignore_index=True)
    # Retain all episode raw signals separately; this index/summary never manufactures missing future labels.
    table(R/'map_identity_verification.csv',[dict(scene=s,road_geometry_sha256=next(iter(v))) for s,v in signatures.items()])
    keys=['success','crash','out_of_road','route_completion','timeout','episode_return','discounted_return','episode_length','average_speed','safety_cost','replacement_rate',
        'comfort_index','steering_rate_rms_raw','steering_rate_p95_raw','steering_jerk_rms_raw','longitudinal_jerk_rms_filtered','longitudinal_jerk_p95_filtered',
        'lateral_jerk_rms_filtered','lateral_jerk_p95_filtered','lateral_acceleration_rms_filtered','lateral_acceleration_p95_filtered',
        'steering_direction_reversals','hard_braking_events','lane_change_count','lane_change_reversal','lateral_oscillation_count','reference_tracking_rms']
    allstats={};comparisons=[]
    for left,right in [('B','A'),('C','A'),('C','B')]:
        l=frame[frame.configuration==left].set_index(['scene','seed']);r=frame[frame.configuration==right].set_index(['scene','seed'])
        diff=(l[keys]-r[keys]).reset_index()
        for key in ['comfort_index','steering_rate_rms_raw','longitudinal_jerk_rms_filtered','lateral_jerk_rms_filtered','discounted_return']:
            diff[key+'_relative']=(l[key]-r[key]).to_numpy()/r[key].abs().to_numpy()
        stats=clustered(diff,[x for x in diff if x not in ['scene','seed']]);stats['per_seed']=diff.groupby('seed').mean(numeric_only=True).to_dict('index')
        allstats[left+'-'+right]=stats
        for key,v in stats.items():
            if key!='per_seed':comparisons.append(dict(comparison=left+'-'+right,metric=key,mean=v['mean'],ci_low=v['ci95'][0],ci_high=v['ci95'][1]))
    table(R/'paired_comparisons.csv',comparisons)
    summary=frame.groupby('configuration')[keys].mean();summary.to_csv(R/'summary.csv',mode='x')
    frame.groupby(['scene','configuration'])[keys].mean().to_csv(R/'map_level_statistics.csv',mode='x')
    dump(R/'paired_statistics.json',dict(map_clusters=12,cases=36,episodes=108,bootstrap_samples=2000,bootstrap_seed=20261120,comparisons=allstats))
    lat=[]
    isolated=pd.concat([pd.read_csv(f) for f in (R/'latency_episodes').glob('*/steps.csv')],ignore_index=True)
    for scope,data in [('shared_6worker',step),('isolated_single_worker_warm',isolated[isolated.step>5])]:
        for group,groupdata in data.groupby('configuration'):
            for key in [k for k in groupdata if k.endswith('_ms')]:
                v=pd.to_numeric(groupdata[key],errors='coerce').dropna()
                if len(v):lat.append(dict(configuration=group,scope=scope,stage=key,samples=len(v),mean_ms=v.mean(),p50_ms=v.quantile(.5),p95_ms=v.quantile(.95),p99_ms=v.quantile(.99),maximum_ms=v.max()))
    table(R/'latency.csv',lat)
    p95={g:float(isolated[(isolated.configuration==g)&(isolated.step>5)].planner_ms.quantile(.95)) for g in ['A','B','C']}
    ca=allstats['C-A'];cb=allstats['C-B'];bounds=p['acceptance']
    point=dict(comfort_at_least10=ca['comfort_index_relative']['mean']<=-.1,
        success_noninferior=ca['success']['mean']>=bounds['success_noninferiority'],crash_noninferior=ca['crash']['mean']<=bounds['crash_noninferiority'],
        oor_noninferior=ca['out_of_road']['mean']<=bounds['oor_noninferiority'],route_noninferior=ca['route_completion']['mean']>=bounds['route_noninferiority'],
        return_noninferior=ca['discounted_return_relative']['mean']>=bounds['discounted_return_relative_noninferiority'])
    confidence=dict(comfort_at_least10=ca['comfort_index_relative']['ci95'][1]<=-.1,
        steering_better=ca['steering_rate_rms_raw']['ci95'][1]<0,longitudinal_jerk_not_worse=ca['longitudinal_jerk_rms_filtered']['ci95'][1]<=0,
        lateral_jerk_not_worse=ca['lateral_jerk_rms_filtered']['ci95'][1]<=0,lateral_oscillation_not_worse=ca['lateral_oscillation_count']['ci95'][1]<=0,
        success_noninferior=ca['success']['ci95'][0]>=bounds['success_noninferiority'],crash_noninferior=ca['crash']['ci95'][1]<=bounds['crash_noninferiority'],
        oor_noninferior=ca['out_of_road']['ci95'][1]<=bounds['oor_noninferiority'],route_noninferior=ca['route_completion']['ci95'][0]>=bounds['route_noninferiority'],
        return_noninferior=ca['discounted_return_relative']['ci95'][0]>=bounds['discounted_return_relative_noninferiority'],
        beats_simple_filter_comfort_or_task=(cb['comfort_index_relative']['ci95'][1]<0 or
            (cb['comfort_index_relative']['ci95'][1]<=0 and
             (cb['success']['ci95'][0]>0 or cb['discounted_return_relative']['ci95'][0]>0))),
        across_maps=sum(v<0 for v in ca['comfort_index_relative']['maps'].values())>=8,
        nonzero_replacement=float(summary.loc['C','replacement_rate'])>0,
        isolated_planner_under100=p95['C']<=100,isolated_planner_not_materially_above_A=p95['C']<=bounds['planner_ratio_to_A']*p95['A'])
    dump(R/'acceptance.json',dict(engineering_point_conditions=point,map95_conditions=confidence,accepted=all(confidence.values()),
        isolated_planner_p95_ms=p95,conclusion='pass' if all(confidence.values()) else 'insufficient evidence or failed constraints; do not deploy',
        initial_pairings=exactpairs,map_geometry_unique=True,real_comfort_vs_predicted_cost='Actual environment metrics only; comfort scores not driving returns',
        finite_map_uncertainty='12clusters; 36planner seeds are not36 independent maps'))
    print(summary.to_string(),flush=True);print('ACCEPTANCE',confidence,flush=True)
if __name__=='__main__':main()
