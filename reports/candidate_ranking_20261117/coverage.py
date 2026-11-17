"""Coverage/scale diagnostics; never select observations by realized return."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from dataset import build
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump


def describe(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)];return dict(n=len(x),p0=float(np.min(x)),p25=float(np.quantile(x,.25)),p50=float(np.median(x)),p75=float(np.quantile(x,.75)),p95=float(np.quantile(x,.95)),p100=float(np.max(x))) if len(x) else dict(n=0)


def main():
 p=json.loads((R/'protocol.json').read_text());old=Path(p['old_data']);aud=json.loads((R/'data_audit.json').read_text());correlations=[];tokens={};duplicate_windows=0;windows=0
 for row in aud['branches']:
  d=np.load(old/'training_data'/f"{row['scenario']}_{row['before']}"/(row['branch']+'.npz'));y=d['targets']
  if len(y)>2 and np.std(y)>0:correlations.append(float(np.corrcoef(y[:-1],y[1:])[0,1]))
  for start in d['starts']:
   token=hashlib.sha256(d['observations'][start].tobytes()+d['actions'][start:start+10].tobytes()).hexdigest();duplicate_windows+=int(token in tokens);tokens[token]=1;windows+=1
 result=dict(old_coverage=dict(native_branches=69,unavailable_tasks=3,origins=23,fit_origins=16,fit_pairs=48,effective_fit_pairs=sum(x['effective'] and x['scenario'] in p['old_fit_scenarios'] for x in aud['pairs']),all_pair_delta=describe([x['delta_real'] for x in aud['pairs']]),abs_pair_delta=describe([abs(x['delta_real']) for x in aud['pairs']]),positive_delta_fraction=float(np.mean([x['delta_real']>0 for x in aud['pairs']])),near_tie_fraction=float(np.mean([not x['effective'] for x in aud['pairs']])),lag1_rtg_correlation=describe(correlations),exact_duplicate_windows=duplicate_windows,windows=windows,action_perturbation_rmse=describe([x['action_rmse'] for x in aud['branches'] if x['branch']!='B']),crash_branches=sum(x['crash'] for x in aud['branches']),offroad_branches=sum(x['out_of_road'] for x in aud['branches'])))
 for split in ['fit','inner_stop','validation']:
  d=build(split);items=d['items'];groups=d['groups'];scene_rows=[];deltas=[];rmse=[];per_origin=[];coverage=[];within_separation=[];prediction_error=[];between_returns=[];baseline_deltas=[];branch_gains={}
  for group in groups:
   first=items[group[0]];base=next(i for i in group if items[i]['branch'] in ('B','baseline'));yy=np.array([items[i]['target'] for i in group]);between_returns.append(float(yy.mean()))
   local=[]
   for i in group:
    if i!=base:
     gain=items[i]['target']-items[base]['target'];baseline_deltas.append(gain);branch_gains.setdefault(items[i]['branch'],[]).append(gain)
   for a,i in enumerate(group):
    for j in group[a+1:]:
     real=items[i]['target']-items[j]['target'];deltas.append(real);local.append(abs(real)>p['pair_tolerance']);within_separation.append(float(np.linalg.norm(items[i]['terminal']-items[j]['terminal'])))
   rmse.extend([float(np.mean((items[i]['actions'][:10]-items[base]['actions'][:10])**2)**.5) for i in group if i!=base]);per_origin.append(dict(scene=first['scene'],before=first['before'],candidates=len(group),pairs=len(local),effective=sum(local),within_return_std=float(yy.std())))
   if first['source']=='new':
    folder=R/'data'/f"{first['scene']}_{first['before']}";origin=json.loads((folder/'baseline_origin.json').read_text());coverage.append(dict(scene=first['scene'],before=first['before'],**origin['coverage']))
   for i in group:
    if np.isfinite(items[i]['actual_terminal']).all():prediction_error.append(float(np.linalg.norm(items[i]['terminal']-items[i]['actual_terminal'])))
  for s in sorted(set(x['scene'] for x in items)):
   rr=[x for x in per_origin if x['scene']==s];scene_rows.append(dict(scene=s,origins=len(rr),pairs=sum(x['pairs'] for x in rr),effective_pairs=sum(x['effective'] for x in rr)))
  meta=[]
  for i in items:
   if i['source']=='new':meta.append(json.loads((R/'data'/f"{i['scene']}_{i['before']}"/(i['branch']+'_meta.json')).read_text()))
  result[split]=dict(scenes=len(scene_rows),origins=len(groups),raw_branches=len(items),deduplicated=d['deduplicated'],pairs=len(d['pairs']),effective_pairs=sum(x['effective_pairs'] for x in scene_rows),unavailable=d['unavailable'],per_scene=scene_rows,per_origin=per_origin,all_pair_delta=describe(deltas),abs_pair_delta=describe(np.abs(deltas)),candidate_vs_baseline=describe(baseline_deltas),positive_vs_baseline_fraction=float(np.mean(np.array(baseline_deltas)>p['pair_tolerance'])),negative_vs_baseline_fraction=float(np.mean(np.array(baseline_deltas)<-p['pair_tolerance'])),branch_gain_counts={b:dict(count=len(g),better=int((np.array(g)>p['pair_tolerance']).sum()),worse=int((np.array(g)<-p['pair_tolerance']).sum()),near=int((np.abs(g)<=p['pair_tolerance']).sum())) for b,g in branch_gains.items()},positive_delta_fraction=float(np.mean(np.array(deltas)>0)),near_tie_fraction=float(np.mean(np.abs(deltas)<=p['pair_tolerance'])),action_rmse=describe(rmse),within_return_std=describe([x['within_return_std'] for x in per_origin]),between_origin_return_std=float(np.std(between_returns)),within_candidate_terminal_l2=describe(within_separation),imagined_vs_actual_terminal_l2=describe(prediction_error),coverage=coverage,new_branch_crashes=sum(x['crash'] for x in meta),new_branch_oor=sum(x['out_of_road'] for x in meta),new_branch_success=sum(x['success'] for x in meta),near_zero_constant_features=None)
  # Scale and redundancy diagnostics are fitted ONLY on training maps.
  if split=='fit':
   w=d['sample_weight'].numpy();w=w/w.sum();z0=d['z0'].numpy();z=d['terminal'].numpy();a=d['actions'][:,:10].flatten(1).numpy();x=np.concatenate([z0,z,a],1);mu=(w[:,None]*x).sum(0);sd=(w[:,None]*(x-mu)**2).sum(0)**.5
   result['fit']['feature_scales']=dict(z0_std=describe(sd[:512]),terminal_std=describe(sd[512:1024]),actions_std=describe(sd[1024:]),near_constant_count=int((sd<.001).sum()),context_terminal_coordinate_correlation=describe([np.corrcoef(z0[:,j],z[:,j])[0,1] for j in range(512) if np.std(z0[:,j])>1e-6 and np.std(z[:,j])>1e-6]),note='Statistics/normalization are fit-only; z0 is repeated within origin. Repeated origins and correlated windows never create extra independent maps.')
 dump(R/'data_coverage.json',result)
if __name__=='__main__':main()
