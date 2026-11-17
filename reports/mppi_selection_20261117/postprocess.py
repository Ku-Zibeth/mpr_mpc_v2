"""Complete paired factorial contrasts and trace-level causal/logging checks."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from evaluation_statistics import cluster_intervals
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump
p=json.loads((R/'protocol.json').read_text());data=json.loads((R/'stage2_analysis.json').read_text());k=json.loads((R/'iteration_choice.json').read_text())['iterations'];ref=f'M{k}';by={(r['configuration'],r['scenario'],r['test_seed']):r for r in data['rows']};contrasts={};comparisons=[('M1','M2'),('M1','M4'),('M2','M4'),(ref,'P'),(ref,'E'),('P','PE'),('E','PE'),(ref,'PE')]
for reference,candidate in comparisons:
 rows=[]
 for scene in p['development_scenarios']:
  for seed in p['repeat_seeds']:
   a=by[(reference,scene,seed)];b=by[(candidate,scene,seed)];rows.append(dict(scenario=scene,test_seed=seed,**{key+'_difference':b[key]-a[key] for key in ['success','crash','out_of_road','route_completion','episode_return','actual_discounted_return','first_action_correction','baseline_selected_rate']}))
 metrics={key:(key,None) for key in rows[0] if key not in ['scenario','test_seed']};contrasts[candidate+'_minus_'+reference]=dict(rows=rows,summary=cluster_intervals(rows,metrics,samples=p['bootstrap_samples'],seed=p['bootstrap_seed']))
checks=[]
for scene in p['development_scenarios']:
 for seed in p['repeat_seeds']:
  directory=R/'episodes/development';tr={g:np.load(directory/f'{g}_{scene}_r{seed}'/'trace.npz') for g in [ref,'P','E','PE']}
  for a,b in [(ref,'E'),('P','PE')]:
   check=dict(scenario=scene,test_seed=seed,reference=a,candidate=b,first_cycle_search_actions_equal=np.array_equal(tr[a]['iteration_actions'][0],tr[b]['iteration_actions'][0]),first_cycle_search_values_equal=np.array_equal(tr[a]['iteration_values'][0],tr[b]['iteration_values'][0]),first_cycle_Q_members_equal=np.array_equal(tr[a]['iteration_q_members'][0],tr[b]['iteration_q_members'][0]));checks.append(check);assert all(check[key] for key in ['first_cycle_search_actions_equal','first_cycle_search_values_equal','first_cycle_Q_members_equal'])
  assert np.array_equal(tr[ref]['iteration_actions'][0,0,:,:10],tr['P']['iteration_actions'][0,0,:,:10])
  for g in ['P','PE']:
   x=tr[g]['iteration_actions'];assert np.all(x[:,:,:,-1]==x[:,:,0:1,-1])
for row in data['rows']:
 f=R/'episodes/development'/f"{row['configuration']}_{row['scenario']}_r{row['test_seed']}"/'trace.npz';tr=np.load(f);n=len(tr['actions']);assert row['counts']['lattice_prepare']==n and row['counts']['mppi']==(n if row['options']['mppi'] else 0);assert len(tr['observations'])==n+1 and bool(tr['done'][-1]) and not tr['done'][:-1].any()
 if row['options']['mppi']:
  assert tr['iteration_actions'].shape[1:]==(row['options']['iterations'],65,11,2);source=tr['selected_source'];selected=tr['final_candidates'][np.arange(n),source,0];assert np.array_equal(selected,tr['actions']);assert np.all((source==0)|(tr['final_values'][np.arange(n),source]>tr['final_values'][:,0]));assert tr['final_q_members'].shape==(n,2)
with (R/'paired_returns.csv').open('w',newline='') as f:
 columns=['configuration','scenario','test_seed','reference']+list(next(iter(data['paired_rows'].values()))[0]);columns=list(dict.fromkeys(columns));writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
 for group,rows in data['paired_rows'].items():
  for r in rows:writer.writerow(dict(configuration=group,reference='B',**r))
dump(R/'factorial_contrasts.json',dict(iterations=k,contrasts=contrasts,scope='Paired whole-native-episode effects, same maps/seeds; terminal/selection comparisons at the single preselected iteration count. No endpoint replay or re-selection by future returns.'))
dump(R/'trace_verification.json',dict(first_cycle_checks=checks,all56_native_episodes_validated=True,all_control_ticks_replanned=True,all_selected_first_actions_match=True,all_nonbaseline_selections_strictly_better_predicted=True,complete_candidate_arrays_saved=True,policy_prior_padding_fixed=True,initial_common_rng_matched=data['initial_state_RNG_pairs_matched']))
