"""Scene-clustered paired comparisons; development-only choices before validation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from evaluation_statistics import cluster_intervals
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump
KEYS=['success','crash','out_of_road','route_completion','episode_return','actual_discounted_return','safety_cost','baseline_selected_rate','first_action_correction','selected_predicted_gain','planning_time_ms','planning_p95_ms']


def analyze(phase,groups,p):
 rows=[];origins={}
 for scene in p[phase+'_scenarios']:
  for seed in p['repeat_seeds']:
   ref=None
   for group in groups:
    folder=R/'episodes'/phase/f'{group}_{scene}_r{seed}';row=json.loads((folder/'episode.json').read_text());assert row['native_done'];origin=json.loads((folder/'origin.json').read_text())
    if ref is None:ref=origin
    else:assert ref==origin,(scene,seed,group,'Initial observed/RNG state mismatch')
    rows.append(row)
 summary={g:cluster_intervals([r for r in rows if r['configuration']==g],{k:(k,None) for k in KEYS},samples=p['bootstrap_samples'],seed=p['bootstrap_seed']) for g in groups};base={(r['scenario'],r['test_seed']):r for r in rows if r['configuration']=='B'};comparisons={};paired={};seed_gains={}
 for group in groups:
  if group=='B':continue
  rr=[]
  for row in [r for r in rows if r['configuration']==group]:
   b=base[(row['scenario'],row['test_seed'])];rr.append(dict(scenario=row['scenario'],test_seed=row['test_seed'],**{key+'_difference':row[key]-b[key] for key in ['success','crash','out_of_road','route_completion','episode_return','actual_discounted_return','safety_cost','planning_time_ms']}))
  comparisons[group]=cluster_intervals(rr,{key:(key,None) for key in rr[0] if key not in ['scenario','test_seed']},samples=p['bootstrap_samples'],seed=p['bootstrap_seed']);paired[group]=rr;seed_gains[group]={str(seed):float(np.mean([r['actual_discounted_return_difference'] for r in rr if r['test_seed']==seed])) for seed in p['repeat_seeds']}
 return dict(phase=phase,summary=summary,comparisons=comparisons,paired_rows=paired,seed_return_gains=seed_gains,rows=rows,initial_state_RNG_pairs_matched=True,native_closed_loop=True)


def eligible(group,result,require_seeds):
 r=result['summary'][group];b=result['summary']['B'];return bool(r['success']>b['success'] and r['crash']<=b['crash'] and r['out_of_road']<=b['out_of_road'] and r['actual_discounted_return']>b['actual_discounted_return'] and (not require_seeds or all(x>0 for x in result['seed_return_gains'][group].values())))


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['stage1','stage2','validation'],required=True);a=ap.parse_args();p=json.loads((R/'protocol.json').read_text())
 if a.stage=='stage1':
  result=analyze('development',['B','M1','M2','M4'],p);allowed=[g for g in ['M1','M2','M4'] if eligible(g,result,False)];pool=allowed or ['M1','M2','M4'];chosen=max(pool,key=lambda g:(result['summary'][g]['actual_discounted_return']-result['summary']['B']['actual_discounted_return'],-p['stage1'][g]['iterations']));choice=dict(iterations=p['stage1'][chosen]['iterations'],group=chosen,eligible=allowed,diagnostic_only=not bool(allowed),selection_source='Development ONLY; validation unread. No online default changed.');assert not (R/'iteration_choice.json').exists();dump(R/'iteration_choice.json',choice);print('Iteration choice',choice,flush=True)
 elif a.stage=='stage2':
  result=analyze('development',['B','M1','M2','M4','P','E','PE'],p);allowed=[g for g in result['summary'] if g!='B' and eligible(g,result,True)];k=json.loads((R/'iteration_choice.json').read_text())['iterations'];chosen=max(allowed,key=lambda g:(result['summary'][g]['actual_discounted_return']-result['summary']['B']['actual_discounted_return'],result['summary'][g]['success'],-p['stage1'][g]['iterations'] if g in p['stage1'] else -k)) if allowed else None
  selection=dict(candidate=chosen,eligible=allowed,iteration_choice=k,validation_opened=False,online_enable=False,reason='One development-selected candidate may undergo exactly one paired validation.' if chosen else 'No method improves development success/return at both seeds without accident worsening. Keep validation sealed; no online selection change.');assert not (R/'development_selection.json').exists();dump(R/'development_selection.json',selection);print(selection,flush=True)
 else:
  choice=json.loads((R/'development_selection.json').read_text());assert choice['candidate'];result=analyze('validation',['B',choice['candidate']],p);group=choice['candidate'];comp=result['comparisons'][group];rr=result['paired_rows'][group];loso={str(scene):float(np.mean([r['actual_discounted_return_difference'] for r in rr if r['scenario']!=scene])) for scene in p['validation_scenarios']};passed=eligible(group,result,True) and comp['actual_discounted_return_difference_ci95_low']>0 and all(x>0 for x in loso.values());dump(R/'validation_decision.json',dict(candidate=group,driving_criteria_passed=bool(passed),leave_one_scene_out_return_gain=loso,latency_gate_pending=True,online_enable=False,interpretation='A single held-out evaluation; no switch/retraining/extra parameter search after this result.'))
 path=R/(a.stage+'_analysis.json');assert not path.exists();dump(path,result);print({g:{k:r[k] for k in ['success','crash','out_of_road','actual_discounted_return']} for g,r in result['summary'].items()},flush=True)
if __name__=='__main__':main()
