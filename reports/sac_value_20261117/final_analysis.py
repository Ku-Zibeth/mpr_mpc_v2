"""Matched four-candidate outcomes; never promote a head from predicted scores alone."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'))
from analyze import pair_accuracy,correlation
from probe import dump
from evaluation_statistics import cluster_intervals
from tools.value_calibration import assert_origin_match
import numpy as np,torch
p=json.loads((R/'protocol.json').read_text());rows=[];origins=[];replay=[]
for s,before in p['targeted_dev_origins']:
 folder=R/'development_counterfactual'/f'{s}_{before}';ref=json.loads((folder/'baseline_origin.json').read_text());audit=json.loads((folder/'audit.json').read_text());case=[]
 for name in ('baseline','mean','best_elite','final_elite'):
  assert_origin_match(ref,json.loads((folder/(name+'_origin.json')).read_text()));data=json.loads((folder/(name+'.json')).read_text());case.append(data)
  rows.append(dict(scenario=s,before=before,choice=name,success=float(data['success']),crash=float(data['crash']),out_of_road=float(data['out_of_road']),route_completion=data['route_completion'],
    episode_return=data['episode_return'],actual_return=data['actual_return'],average_safety_cost=data['cost'],planning_time_ms=data['planning_time_ms'],predicted_return=data['predicted_return'],
    prediction_mae=abs(data['predicted_return']-data['actual_return'])))
 samples=np.load(folder/'audit_samples.npz');assert samples['actions'].shape==(258,11,2) and samples['iteration_actions'].shape==(4,64,11,2)
 assert audit['metrics']['wm_evaluated_candidates']==262 and audit['definition']['terminal_value']=='sac_value'
 old=R/'development_counterfactual_failed_logging'/f'{s}_{before}'/'audit_samples.npz'
 if old.exists():
  old=np.load(old);assert np.array_equal(samples['actions'],old['actions']);replay.append(dict(scenario=s,before=before,failed_logging_attempt_action_max_error=0.))
 mx=np.array([x['position'] for x in case[1]['trace'][:10]]);ex=np.array([x['position'] for x in case[2]['trace'][:10]]);n=min(len(mx),len(ex))
 elite=samples['original_elites'][:,:,:10].reshape(32,-1);dist=np.linalg.norm(elite[:,None]-elite[None,:],axis=-1);ii,jj=np.triu_indices(32,1)
 orig=dict(scenario=s,before=before,mean_best_action_l2=float(np.linalg.norm(np.array(case[1]['actions'])[:10]-np.array(case[2]['actions'])[:10])),
   mean_best_true_xy_rmse_m=float(np.sqrt(np.mean(np.sum((mx[:n]-ex[:n])**2,axis=1)))),elite_pairwise_action_l2=float(dist[ii,jj].mean()),
   elite_positive_first_steer_fraction=float((elite[:,0]>0).mean()),unique_prefixes=audit['unique_prefixes'],
   prediction_ranking_accuracy=pair_accuracy([x['predicted_return'] for x in case],[x['actual_return'] for x in case]),
   prediction_spearman=correlation([x['predicted_return'] for x in case],[x['actual_return'] for x in case],True),
   predicted_order=[x['choice'] for x in sorted(case,key=lambda x:x['predicted_return'],reverse=True)],real_order=[x['choice'] for x in sorted(case,key=lambda x:x['actual_return'],reverse=True)],
   predicted_mean_minus_best=case[1]['predicted_return']-case[2]['predicted_return'],real_mean_minus_best=case[1]['actual_return']-case[2]['actual_return'])
 origins.append(orig)
metrics=('success','crash','out_of_road','route_completion','episode_return','actual_return','average_safety_cost','planning_time_ms','predicted_return','prediction_mae')
summary=[dict(choice=name,**cluster_intervals([x for x in rows if x['choice']==name],{k:(k,None) for k in metrics},samples=2000,seed=1)) for name in ('baseline','mean','best_elite','final_elite')]
by={x['choice']:x for x in summary};benefit=by['mean']['actual_return']>by['baseline']['actual_return'] and by['mean']['episode_return']>by['baseline']['episode_return'] and by['mean']['success']>=by['baseline']['success'] and by['mean']['crash']<=by['baseline']['crash'] and by['mean']['out_of_road']<=by['baseline']['out_of_road']
value=json.loads((R/'existing_pool_evaluation.json').read_text())['validation'];decision=dict(value_ranking_passed=value['passed'],development_mean_improves=bool(benefit),worth_independent_closed_loop=bool(value['passed'] and benefit),default_output_unchanged=True,
 cost_weight=0.,formal100=False,reason='No reliable predicted-latent value ranking; do not deploy or expand evaluation.' if not value['passed'] else 'Value ranking passed but independent driving benefit is not yet established.')
dump(R/'development_comparison.json',dict(summary=summary,rows=rows,origins=origins,replay=replay,decision=decision,
 scope='Four development origins, native H-prefix then original SAC suffix, NOT full receding-horizon MPPI episodes. Cost means sum of true safety cost after origin. Planning ms is LocalMPPI plus audit synchronized wall time, excludes preceding Lattice/SAC context creation. Predicted winner not automatically executed online. Positive/negative steer differences do not prove discrete driving modes.'))
for x in summary:print(x['choice'],{k:x[k] for k in metrics})
print(decision)
