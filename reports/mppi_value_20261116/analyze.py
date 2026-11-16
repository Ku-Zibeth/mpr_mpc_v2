"""Paired bootstrap summaries and predeclared, provisional selection only."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));REPORT=Path(__file__).parent
from evaluation_statistics import summarize_runs,paired_comparisons
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validation','test'],default='validation');a=p.parse_args()
meta=json.loads((REPORT/f'{a.phase}_run.json').read_text());out=Path(meta['output']);rows=[]
for group in meta['groups']:
 part=[]
 for row in csv.DictReader((out/group/'episodes.csv').open()):
  for key,value in list(row.items()):
   try:row[key]=float(value) if value else None
   except (ValueError,TypeError):pass
  row['configuration']=group;part.append(row)
 assert len(part)==len(meta['scenarios']) and {int(r['scenario']) for r in part}==set(meta['scenarios'])
 assert all(int(r['length'])<=1000 for r in part)
 rows.extend(part)
summary=[summarize_runs(g,[r for r in rows if r['configuration']==g],samples=2000,seed=1) for g in meta['groups']]
pairs=[('B',g) for g in meta['groups'] if g!='B']
if a.phase=='validation':pairs += [('D0',g) for g in meta['groups'] if g not in ('B','D0')]
comp=paired_comparisons(rows,samples=2000,seed=1,pairs=pairs)
lookup={(r['configuration'],int(r['scenario'])):r for r in rows};paired=[]
metrics=['final_success','episode_crash','episode_out_of_road','final_route_completion','return','actual_discounted_return','mean_speed_mps','timeout','mppi_predicted_return_gain','mppi_prefix_reward_gain','mppi_discounted_terminal_gain','mppi_first_throttle_delta','mppi_first_action_delta_norm','mppi_weight_entropy','mppi_effective_sample_size','mppi_sampling_diversity','mean_mpr_total_planning_ms']
for b,d in pairs:
 for scenario in meta['scenarios']:
  rb,rd=lookup[b,scenario],lookup[d,scenario];r=dict(comparison=f'{d} - {b}',scenario=scenario,test_seed=1)
  for key in metrics:
   r[key+'_baseline']=rb.get(key);r[key+'_candidate']=rd.get(key)
   try:r[key+'_delta']=rd[key]-rb[key]
   except (TypeError,KeyError):r[key+'_delta']=None
  paired.append(r)
with (out/'paired_scenarios.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
with (out/'episodes.csv').open('w',newline='') as f:
 keys=sorted(set().union(*(r.keys() for r in rows)));writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
behavior=[]
for group in meta['groups']:
 if group=='B':continue
 rr=[r for r in rows if r['configuration']==group]
 def corr(x,y):
  xy=np.asarray(list(zip(x,y)),float);xy=xy[np.isfinite(xy).all(1)]
  return float(np.corrcoef(xy.T)[0,1]) if len(xy)>1 and np.std(xy[:,0])>0 and np.std(xy[:,1])>0 else None
 delta=[r['actual_discounted_return']-lookup['B',int(r['scenario'])]['actual_discounted_return'] for r in rr]
 behavior.append(dict(configuration=group,episodes=len(rr),
  throttle_vs_actual_return_delta_correlation=corr([r['mppi_first_throttle_delta'] for r in rr],delta),
  correction_vs_actual_return_delta_correlation=corr([r['mppi_first_action_delta_norm'] for r in rr],delta),
  predicted_vs_actual_return_delta_correlation=corr([r['mppi_predicted_return_gain'] for r in rr],delta),
  throttle_vs_crash_correlation=corr([r['mppi_first_throttle_delta'] for r in rr],[r['episode_crash'] for r in rr]),
  throttle_vs_offroad_correlation=corr([r['mppi_first_throttle_delta'] for r in rr],[r['episode_out_of_road'] for r in rr]),
  note='Exploratory episode-level correlation on a small sample, not causality or a new score/gate.'))
result=dict(phase=a.phase,summary=summary,paired_bootstrap95=comp,behavior_association=behavior,bootstrap_samples=2000,bootstrap_seed=1,
 note='Scenario-clustered paired percentile bootstrap, small sample; 95% intervals do not correct multiple comparisons. Shared-load timing; not a real-time benchmark. Positive predicted gains do not establish causality or actual benefit.')
(out/'analysis.json').write_text(json.dumps(result,indent=2));(REPORT/f'{a.phase}_summary.json').write_text(json.dumps(result,indent=2))
if a.phase=='validation':
 by={r['configuration']:r for r in summary};b=by['B'];eligible=[g for g in ('D1','D2','D3','D4') if by[g]['success']>=b['success'] and by[g]['crash']<=b['crash'] and by[g]['out_of_road']<=b['out_of_road'] and by[g]['reward']>b['reward']]
 eligible.sort(key=lambda g:(by[g]['success'],by[g]['reward']),reverse=True)
 selection=dict(provisional_candidate=eligible[0] if eligible else None,eligible=eligible,criterion=meta['protocol']['selection'],
  reason='Predeclared point-estimate screening, subject to sealed test and uncertainty.' if eligible else 'No updated variant satisfies every predeclared criterion. No combination, sealed test or final100 is justified.',
  configuration_default='D1 (requested inverse-beta correction); not a claim of empirical optimum',old100_not_used_for_selection=True)
 (REPORT/'selection.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection,indent=2))
for r in summary:print(r['configuration'],{k:r[k] for k in ['success','crash','out_of_road','route_completion','reward','actual_discounted_return','planning_time_ms']})
