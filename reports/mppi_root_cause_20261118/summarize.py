"""Map-cluster paired statistics for the preregistered native closed-loop experiment."""
from collections import defaultdict
import json
import numpy as np
import pandas as pd
from study_io import R,load,dump,table

METRICS=['success','crash','out_of_road','route_completion','timeout','episode_return','discounted_return','episode_length','safety_cost','baseline_selected_rate','first_action_delta']

def ci(values):
    v=np.asarray(values,float);rng=np.random.default_rng(20261118)
    return dict(mean=float(v.mean()),ci95=np.quantile(v[rng.integers(len(v),size=(2000,len(v)))].mean(1),[.025,.975]).tolist())

def main():
    assert (R/'probe_complete.json').exists()
    protocol=load(R/'protocol.json');episodes=[];coverage=[];alignment=[];steps=[];replanning=[];probes=[]
    bykey={};checks=[]
    for folder in sorted((R/'episodes').iterdir()):
        if not (folder/'episode.json').exists():continue
        row=load(folder/'episode.json');group=row['configuration'];scene=row['scene'];seed=row['repeat_seed']
        episodes.append({k:v for k,v in row.items() if k not in ['initial_hashes','counts']});bykey[group,scene,seed]=row
        s=pd.read_csv(folder/'steps.csv');steps.append(s)
        if group!='B':coverage.append(pd.read_csv(folder/'coverage.csv'))
        alignment.append(pd.read_csv(folder/'alignment.csv'))
        z=np.load(folder/'trace.npz');rewards=z['rewards'];rtg=z['rtg'];controls=z['actions'];seq=z['selected_sequences']
        assert len(rewards)==row['episode_length'] and len(rtg)==len(rewards)+1 and z['done'][-1]
        assert not z['done'][:-1].any() and np.isfinite(controls).all()
        assert np.array_equal(seq[:,0],controls)
        assert np.allclose(rtg[:-1],rewards+.995*rtg[1:],rtol=0,atol=1e-8)
        assert row['weights_unchanged'] and row['native_done']
        checks.append(dict(episode=folder.name,first_action_exact=True,native_done=True,rtg_verified=True,weights_unchanged=True))
        # Difference between the old plan's future controls and controls freshly replanned/executed later.
        for h in [1,2,4,10]:
            if len(controls)>h:
                delta=seq[:-h,h]-controls[h:]
                replanning.append(dict(configuration=group,scene=scene,repeat_seed=seed,horizon=h,matched_steps=len(delta),
                    planned_vs_replanned_action_l2=float(np.linalg.norm(delta,axis=1).mean()),
                    steering_mae=float(np.abs(delta[:,0]).mean()),throttle_mae=float(np.abs(delta[:,1]).mean()),
                    protocol='native rolling control; old future plan vs later newly replanned actual control; no fixed-H replay'))
    assert len(episodes)==48 and all((g,s,seed) in bykey for g in ['B','C','D'] for s in protocol['scenarios'] for seed in [1,2])
    origin_checks=[]
    for scene in protocol['scenarios']:
        for seed in [1,2]:
            origin=[bykey[g,scene,seed]['initial_hashes'] for g in ['B','C','D']]
            assert origin[0]==origin[1]==origin[2],(scene,seed)
            origin_checks.append(dict(scene=scene,repeat_seed=seed,observation_vehicle_traffic_rng_exact=True))
    table('paired_closed_loop_episodes.csv',episodes);table('replanning_action_deviation.csv',replanning)
    c=pd.concat(coverage,ignore_index=True);c.to_csv(R/'candidate_coverage.csv',index=False,mode='x')
    a=pd.concat(alignment,ignore_index=True);a.to_csv(R/'action_alignment.csv',index=False,mode='x')
    s=pd.concat(steps,ignore_index=True);s.to_csv(R/'closed_loop_prediction_steps.csv',index=False,mode='x')
    e=pd.DataFrame(episodes);statistics={};summary=[]
    for g in ['B','C','D']:
        rows=e[e.configuration==g];st=s[s.configuration==g]
        p95=float(st.act_wall_ms.quantile(.95));p50=float(st.act_wall_ms.quantile(.5))
        summary.append(dict(configuration=g,comparison='absolute mean',maps=8,episodes=16,**{k:float(rows[k].mean()) for k in METRICS},planning_p50_ms=p50,planning_p95_ms=p95,planning_max_ms=float(st.act_wall_ms.max())))
    for group,base in [('C','B'),('D','B'),('D','C')]:
        diff=e[e.configuration==group].set_index(['scene','repeat_seed'])[METRICS]-e[e.configuration==base].set_index(['scene','repeat_seed'])[METRICS]
        stat={k:ci(diff[k].groupby('scene').mean().values) for k in METRICS}
        stat['per_scene_discounted_difference']=diff.discounted_return.groupby('scene').mean().to_dict()
        stat['per_seed_discounted_difference']=diff.discounted_return.groupby('repeat_seed').mean().to_dict()
        vals=diff.discounted_return.groupby('scene').mean().values
        stat['leave_one_map_out_gain_range']=[float(np.delete(vals,i).mean()) for i in range(len(vals))]
        statistics[f'{group}-{base}']=stat
        summary.append(dict(configuration=group,comparison=f'paired difference vs {base}',maps=8,episodes=16,**{k:v['mean'] for k,v in stat.items() if k in METRICS}))
    table('summary.csv',summary);dump('paired_statistics.json',dict(unit='8 complete maps;2 paired repeat seeds/map',bootstrap_samples=2000,seed=20261118,comparisons=statistics))
    for file in sorted(R.glob('probe_*.csv')):probes.append(pd.read_csv(file))
    for scene in protocol['scenarios'][:2]:
        states=[load(R/f'probe_{g}_{scene}.json')['physical_state']['hashes'] for g in ['B','C','D']]
        assert states[0]==states[1]==states[2],('probe origin mismatch',scene)
    probe=pd.concat(probes,ignore_index=True);probe.to_csv(R/'latency_probe_samples.csv',index=False,mode='x')
    timing=[]
    for scope,frame,total in [('shared_native_recording',s,'act_wall_ms'),('isolated_frozen_state',probe,'total_wall_ms')]:
        for g,rows in frame.groupby('configuration'):
            for key in [total,*[k for k in rows.columns if k.endswith('_ms') and k!=total]]:
                v=pd.to_numeric(rows[key],errors='coerce').dropna()
                if len(v):timing.append(dict(configuration=g,scope=scope,stage=key,samples=len(v),p50_ms=float(v.quantile(.5)),p95_ms=float(v.quantile(.95)),p99_ms=float(v.quantile(.99)),maximum_ms=float(v.max()),mean_ms=float(v.mean())))
    table('latency.csv',timing)
    coverage_metrics=['unique_candidate_ratio','unique_first_action_ratio','first_action_distance_mean','full_sequence_rmse','clipping_ratio','throttle_clipping','sac_near_first_overlap','elite_diversity','input_std_steer','input_std_throttle']
    cov=[]
    for (g,i),rows in c.groupby(['configuration','iteration']):
        cov.append(dict(configuration=g,iteration=int(i),planning_steps=len(rows),**{k:float(rows[k].mean()) for k in coverage_metrics}))
    table('candidate_coverage_summary.csv',cov)
    effects={}
    for g in ['B','C','D']:
        st=s[s.configuration==g];al=a[a.configuration==g]
        ef=dict(lattice_feasible_mean=float(st.lattice_feasible.mean()),lattice_distinct_parameter_modes_mean=float(st.feasible_parameter_modes.mean()),
            clipped_endpoint_step_ratio=float((st.clipped_future_times>0).mean()),
            submitted_control_max_error=float(al.normalized_control_execution_error.max()),
            next_position_mae=float(al.virtual_next_position_error.mean()),next_speed_mae=float(al.virtual_next_speed_error.abs().mean()),
            next_heading_mae=float(al.virtual_next_heading_error.abs().mean()),old_path_next_control_delta_mean=float(al.old_path_next_control_delta.mean()))
        if g!='B':
            gain=st.selected_predicted_gain;term=st.selected_discounted_terminal_gain
            ef.update(predicted_positive_gain_ratio=float((gain>0).mean()),predicted_gain_mean=float(gain.mean()),
                terminal_gain_absolute_share=float(term.abs().sum()/(term.abs().sum()+st.selected_prefix_gain.abs().sum())),
                first_reward_mae=float((st.predicted_first_reward-st.actual_first_reward).abs().mean()),
                predicted_total_vs_rolling_actual_mae=float((st.predicted_total_return-st.actual_replanning_policy_rtg).abs().mean()))
        effects[g]=ef
    # Repeat seeds can give identical B trajectories; retain them but do not count them as maps.
    repeat_duplicates={g:sum(np.array_equal(np.load(R/'episodes'/f'{g}_{scene}_r1'/'trace.npz')['actions'],np.load(R/'episodes'/f'{g}_{scene}_r2'/'trace.npz')['actions']) for scene in protocol['scenarios']) for g in ['B','C','D']}
    dump('control_chain_statistics.json',dict(groups=effects,identical_repeat_trajectories=repeat_duplicates,
        warning='Ideal-state discrepancies are not interchangeable with WM latent error. Rolling-policy RTG is not H10->SAC counterfactual.'))
    dump('collection_verification.json',dict(episodes=checks,paired_initial_states=origin_checks,episode_count=48,map_count=8,training=0,formal100=False,all_checks_passed=True))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
