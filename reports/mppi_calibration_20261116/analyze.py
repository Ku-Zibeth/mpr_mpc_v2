"""Scene-clustered calibration analysis, never used by the online planner."""
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from tools.value_calibration import assert_origin_match
from evaluation_statistics import cluster_intervals,summarize_runs,paired_comparisons
from probe import clean,dump

def rank(x):
 x=np.asarray(x);order=np.argsort(x);r=np.empty(len(x));i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  r[order[i:j]]=(i+j-1)/2;i=j
 return r

def correlation(a,b,spearman=False):
 x=np.asarray(list(zip(a,b)),float).reshape(-1,2);x=x[np.isfinite(x).all(1)]
 if len(x)<2:return None
 aa,bb=x.T
 if spearman:aa,bb=rank(aa),rank(bb)
 return float(np.corrcoef(aa,bb)[0,1]) if np.std(aa)>0 and np.std(bb)>0 else None

def pair_accuracy(a,b):
 i,j=np.triu_indices(len(a),1);x=np.asarray(a)[i]-np.asarray(a)[j];y=np.asarray(b)[i]-np.asarray(b)[j];valid=abs(y)>1e-6
 return float(np.mean(np.where(abs(x[valid])<=1e-6,.5,np.sign(x[valid])==np.sign(y[valid])))) if valid.any() else None

def probe_analysis(split):
 meta=json.loads((R/'protocol.json').read_text());rows=[];cases=[];errors=[];coverage=[];unavailable=[];elite_comparisons=[]
 paths=sorted((R/'probes').glob(split+'_*'))
 expected=(len(meta[split+'_scenarios'])*2 if split!='diagnosis' else 2);assert len(paths)==expected,(split,len(paths),expected)
 for path in paths:
  missing=list(path.glob('unavailable*.json'))
  if missing:unavailable.append(dict(origin=path.name,branches=[json.loads(x.read_text()) for x in missing]));continue
  ref=json.loads((path/'B_origin.json').read_text());scenario=ref['episode_config']['scenario'];before=ref['episode_config']['before']
  names=['B',*meta['groups']]+(['elite_Q1'] if scenario in meta['elite_scenarios'] else []) if split!='diagnosis' else ['B','legacy_mean']
  branches={name:json.loads((path/(name+'.json')).read_text()) for name in names}
  for name in names:assert_origin_match(ref,json.loads((path/(name+'_origin.json')).read_text()))
  coverage.append(dict(scenario=scenario,before=before,**ref['coverage']))
  if 'elite_Q1' in branches:
   m,e=branches['Q1'],branches['elite_Q1'];pm=m['predicted_prefix']+m['terminal_discount']*m['predicted_q'];pe=e['predicted_prefix']+e['terminal_discount']*e['predicted_q']
   elite_comparisons.append(dict(scenario=scenario,before=before,predicted_mean=pm,predicted_elite=pe,predicted_mean_minus_elite=pm-pe,actual_mean=m['actual_discounted_return'],actual_elite=e['actual_discounted_return'],actual_mean_minus_elite=m['actual_discounted_return']-e['actual_discounted_return']))
  b=branches['B'];real=[r['actual_discounted_return'] for r in branches.values()]
  for name,r in branches.items():
   arr=np.asarray(r['actions'])[:meta['horizon']]-np.asarray(b['actions'])[:meta['horizon']]
   for step in r['prefix']:
    errors.append(dict(scenario=scenario,before=before,branch=name,step=step['step'],
      **{k:step.get(k) for k in ('latent_l2','latent_rmse','reward_open_loop_error','reward_teacher_error','reward_rollout_shift','position_error_m','speed_error_mps','heading_error_rad')}))
   row=dict(scenario=scenario,before=before,branch=name,first_steer_delta=float(arr[0,0]),first_throttle_delta=float(arr[0,1]),sequence_l2=float(np.linalg.norm(arr)),
     sequence_rms=float(np.sqrt(np.mean(arr**2))),latent_rmse=r['prefix'][-1]['latent_rmse'],
     prefix_length=len(r['prefix']),batch_single_latent_max_abs=float(np.max(np.abs(np.asarray(r['predicted_terminal_latent']).reshape(-1)-np.asarray(r['prefix'][-1]['predicted_z_after']).reshape(-1)))) if len(r['prefix'])==meta['horizon'] else None,actual_gain=r['actual_discounted_return']-b['actual_discounted_return'],
     crash=r['crash'],out_of_road=r['out_of_road'],predicted_prefix_gain=r['predicted_prefix']-b['predicted_prefix'],actual_prefix_gain=r['actual_prefix']-b['actual_prefix'])
   teacher=sum(r['gamma']**i*x['reward_teacher'] for i,x in enumerate(r['prefix']));open_r=sum(r['gamma']**i*x['reward_open_loop'] for i,x in enumerate(r['prefix']))
   row.update(reward_head_discounted_error=teacher-r['actual_prefix'],rollout_reward_shift=open_r-teacher,termination_mask_contribution=r['predicted_prefix']-open_r)
   if r['terminal'] is not None:
    t=r['terminal'];row.update(predicted_Q_mean=float(np.mean(t['predicted_z_candidate_q'])),actual_z_candidate_Q_mean=float(np.mean(t['actual_z_candidate_q'])),actual_z_SAC_Q_mean=float(np.mean(t['actual_z_sac_q'])),actual_SAC_RTG=r['actual_continuation'],
     cached_recomputed_Q_gap=float(np.mean(t['predicted_z_candidate_q'])-r['predicted_q']),q_state_shift=np.asarray(t['predicted_z_candidate_q'])-t['actual_z_candidate_q'],sac_q_rtg_difference=np.asarray(t['actual_z_sac_q'])-r['actual_continuation'])
   cases.append(row)
  for label,w in meta['groups'].items():
   predicted=[r['predicted_prefix']+w*r['terminal_discount']*r['predicted_q'] for r in branches.values()]
   index=names.index(label) if split!='diagnosis' else 1;r=branches[names[index]];gain=real[index]-real[0];pg=predicted[index]-predicted[0]
   qsign_real=None;qsign_pred=None;qsign_sac=None;flip=None
   if r['terminal'] and b['terminal']:
    dq=np.asarray(r['terminal']['predicted_z_candidate_q'])-b['terminal']['predicted_z_candidate_q'];aq=np.asarray(r['terminal']['actual_z_candidate_q'])-b['terminal']['actual_z_candidate_q']
    rtg_gain=r['actual_continuation']-b['actual_continuation'];sq=np.asarray(r['terminal']['actual_z_sac_q'])-b['terminal']['actual_z_sac_q']
    if abs(rtg_gain)>1e-6:
     qsign_pred=float(np.mean(np.sign(dq)==np.sign(rtg_gain)));qsign_real=float(np.mean(np.sign(aq)==np.sign(rtg_gain)));qsign_sac=float(np.mean(np.sign(sq)==np.sign(rtg_gain)))
    flip=float(np.mean(np.sign(dq)!=np.sign(aq)))
   rows.append(dict(scenario=scenario,before=before,configuration=label,weight=w,spearman=correlation(predicted,real,True),
    pair_ranking_accuracy=pair_accuracy(predicted,real),top1_regret=max(real)-real[int(np.argmax(predicted))],
    mean_vs_baseline_accuracy=None if abs(gain)<=1e-6 else .5 if abs(pg)<=1e-6 else float(np.sign(gain)==np.sign(pg)),
    actual_gain=gain,predicted_gain=pg,prefix_reward_gain=r['predicted_prefix']-b['predicted_prefix'],
    weighted_terminal_q_gain=w*(r['terminal_discount']*r['predicted_q']-b['terminal_discount']*b['predicted_q']),
    actual_prefix_reward_gain=r['actual_prefix']-b['actual_prefix'],latent_rmse=r['prefix'][-1]['latent_rmse'],
    success=float(r['success']),crash=float(r['crash']),out_of_road=float(r['out_of_road']),
    predicted_state_member_ranking_accuracy=qsign_pred,actual_state_member_ranking_accuracy=qsign_real,actual_state_SAC_action_ranking_accuracy=qsign_sac,state_ranking_flip_fraction=flip))
 summaries=[]
 for label in meta['groups']:
  rr=[x for x in rows if x['configuration']==label];keys=('spearman','pair_ranking_accuracy','top1_regret','mean_vs_baseline_accuracy','actual_gain','predicted_gain','prefix_reward_gain','weighted_terminal_q_gain','actual_prefix_reward_gain','latent_rmse','success','crash','out_of_road','predicted_state_member_ranking_accuracy','actual_state_member_ranking_accuracy','actual_state_SAC_action_ranking_accuracy','state_ranking_flip_fraction')
  summaries.append(dict(configuration=label,origins=len(rr),unique_scenarios=len({x['scenario'] for x in rr}),**cluster_intervals(rr,{k:(k,None) for k in keys},samples=2000,seed=1)))
 associations=[]
 for label in ['Q1','Q05','Q025','Q0'] if split!='diagnosis' else ['legacy_mean']:
  rr=[x for x in cases if x['branch']==label];rank_rows=[x for x in rows if x['configuration']==(label if split!='diagnosis' else 'Q1')]
  associations.append(dict(branch=label,predicted_vs_actual_gain_spearman=correlation([x['predicted_gain'] for x in rank_rows],[x['actual_gain'] for x in rank_rows],True),
   latent_error_vs_ranking_error=correlation([x['latent_rmse'] for x in rank_rows if x['mean_vs_baseline_accuracy'] is not None],[1-x['mean_vs_baseline_accuracy'] for x in rank_rows if x['mean_vs_baseline_accuracy'] is not None],True),sequence_shift_vs_latent_error=correlation([x['sequence_l2'] for x in rr],[x['latent_rmse'] for x in rr],True),
   throttle_vs_crash=correlation([x['first_throttle_delta'] for x in rr],[x['crash'] for x in rr]),throttle_vs_offroad=correlation([x['first_throttle_delta'] for x in rr],[x['out_of_road'] for x in rr]),
   throttle_vs_actual_gain=correlation([x['first_throttle_delta'] for x in rr],[x['actual_gain'] for x in rr],True)))
 per_step=[]
 for name in sorted({x['branch'] for x in errors}):
  for i in range(meta['horizon']):
   rr=[x for x in errors if x['branch']==name and x['step']==i]
   if not rr:continue
   per_step.append(dict(branch=name,step=i,count=len(rr),**{k:float(np.mean([abs(x[k]) for x in rr if x.get(k) is not None])) for k in ('latent_l2','latent_rmse','reward_open_loop_error','reward_teacher_error','reward_rollout_shift','position_error_m','speed_error_mps','heading_error_rad') if any(x.get(k) is not None for x in rr)}))
 dump(R/(split+'_probe_analysis.json'),dict(summary=summaries,rows=rows,cases=cases,per_step_errors=per_step,elite_comparisons=elite_comparisons,associations=associations,coverage=coverage,unavailable=unavailable,
  scope='Scene-clustered bootstrap95%,2000 resamples,seed1; ties in predicted ranking receive half credit, true-return ties excluded. Regret relative to measured pool, not all possible actions. Observed+traffic+RNG match does not certify all hidden simulator state. No latent error is interpreted as meters. Q-vs-SAC RTG includes continuation-policy mismatch. Associations are not causal.'))
 print(split,'origins',len(coverage),'unreachable',len(unavailable),flush=True)
 for x in summaries:print(x['configuration'],{k:x[k] for k in ('actual_gain','mean_vs_baseline_accuracy','pair_ranking_accuracy','spearman','top1_regret')})
 return summaries

def closed_analysis(phase):
 rows=[];meta=json.loads((R/'protocol.json').read_text());scenes=meta['additional_test_scenarios'] if phase=='test' else meta['closed_loop_'+phase+'_scenarios']
 groups=['B',*meta['groups']] if phase!='test' else ['B','Q1',json.loads((R/'selection.json').read_text())['candidate']]
 for g in groups:
  for s in scenes:
   row=next(csv.DictReader((R/'closed_loop'/phase/f'{g}_{s}'/'episodes.csv').open()))
   for k,v in list(row.items()):
    try:row[k]=float(v) if v else None
    except ValueError:pass
   row.update(configuration=g,test_seed=1);rows.append(row)
 summaries=[summarize_runs(g,[x for x in rows if x['configuration']==g],samples=2000,seed=1) for g in groups]
 pairs=[('B',g) for g in groups if g!='B']+[('Q1',g) for g in groups if g not in ('B','Q1')]
 comparisons=paired_comparisons(rows,samples=2000,seed=1,pairs=pairs)
 fields=sorted(set().union(*(x.keys() for x in rows)))
 with (R/(phase+'_episodes.csv')).open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
 # Each row is a single scene; include actual outcomes and action/search diagnostics.
 with (R/(phase+'_paired.csv')).open('w',newline='') as f:
  keys=['comparison','scenario','success_delta','crash_delta','out_of_road_delta','route_delta','return_delta','actual_discounted_return_delta'];wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader()
  by={(x['configuration'],int(x['scenario'])):x for x in rows}
  for b,d in pairs:
   for s in scenes:
    aa,bb=by[b,s],by[d,s];wr.writerow(dict(comparison=d+' - '+b,scenario=s,**{k:bb[src]-aa[src] for k,src in [('success_delta','final_success'),('crash_delta','episode_crash'),('out_of_road_delta','episode_out_of_road'),('route_delta','final_route_completion'),('return_delta','return'),('actual_discounted_return_delta','actual_discounted_return')]}))
 dump(R/(phase+'_closed_analysis.json'),dict(summary=summaries,comparisons=comparisons,scope='Fresh method/scene processes, equal scene seed and config; scene-clustered bootstrap95%,2000 resamples. Timing under shared GPU load is not an isolated performance comparison.'))
 for x in summaries:print(phase,x['configuration'],{k:x[k] for k in ('success','crash','out_of_road','reward','actual_discounted_return')})
 return summaries,comparisons

def select():
 pp=probe_analysis('dev');cc,_=closed_analysis('dev');p={x['configuration']:x for x in pp};c={x['configuration']:x for x in cc};eligible=[]
 base_cases=[x for x in json.loads((R/'dev_probe_analysis.json').read_text())['cases'] if x['branch']=='B'];base_crash=float(np.mean([x['crash'] for x in base_cases]));base_offroad=float(np.mean([x['out_of_road'] for x in base_cases]))
 for g in ('Q05','Q025'):
  if p[g]['actual_gain']<=max(0,p['Q1']['actual_gain']):continue
  if p[g]['crash']>min(base_crash,p['Q1']['crash']) or p[g]['out_of_road']>min(base_offroad,p['Q1']['out_of_road']):continue
  if all(c[g]['success']>=c[b]['success'] and c[g]['reward']>c[b]['reward'] and c[g]['crash']<=c[b]['crash'] and c[g]['out_of_road']<=c[b]['out_of_road'] for b in ('B','Q1')):eligible.append(g)
 eligible.sort(key=lambda g:(c[g]['actual_discounted_return'],c[g]['success']),reverse=True)
 path=R/'selection.json';assert not path.exists();dump(path,dict(candidate=eligible[0] if eligible else None,eligible=eligible,source='Only development probes and closed-loop; no validation or old scenes used.',validation_gate_passed=False));print('selection',eligible)

def validate():
 probe_analysis('validation');cc,comparisons=closed_analysis('validation');path=R/'selection.json';selection=json.loads(path.read_text());g=selection['candidate'];by={x['configuration']:x for x in cc}
 gate=bool(g) and all(by[g]['success']>=by[b]['success'] and by[g]['reward']>by[b]['reward'] and by[g]['crash']<=by[b]['crash'] and by[g]['out_of_road']<=by[b]['out_of_road'] and next(x for x in comparisons if x['comparison']==g+' - '+b)['reward_delta_ci95_low']>0 for b in ('B','Q1'))
 selection.update(validation_gate_passed=gate,reason='Eligible for new fixed test; not a final benefit claim.' if gate else 'No independent, accident-preserving improvement evidence. No new test/full100, no extra coefficients and no retraining.');dump(path,selection);print(selection)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','validation','diagnosis','test'],required=True);a=p.parse_args()
 if a.phase=='dev':select()
 elif a.phase=='validation':validate()
 elif a.phase=='diagnosis':probe_analysis('diagnosis')
 else:closed_analysis('test')
