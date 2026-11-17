"""Audit full old+new cohorts; no selection or tuning using seen validation."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from evaluation_statistics import cluster_intervals
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump


def describe(x):
 x=np.asarray(x,float);return dict(n=len(x),min=float(x.min()),p25=float(np.quantile(x,.25)),p50=float(np.median(x)),p75=float(np.quantile(x,.75)),max=float(x.max())) if len(x) else dict(n=0)


def main():
 p=json.loads((R/'protocol.json').read_text());prior=Path(p['prior_data']);manifest=json.loads((prior/'dataset_manifest.json').read_text());paths={role:[Path(x['path']) for x in manifest[key]] for role,key in [('Fit','fit_sources'),('InnerStop','stop_sources')]};paths['SeenValidation']=[x for x in sorted((prior/'data').glob('*/*.npz')) if x.stem in ['baseline','gaussian','mean','best_elite'] and int(x.parent.name.split('_')[0]) in p['already_seen_validation_scenarios']];report={};all_origins={};allrows=[]
 for role,files in paths.items():
  rows=[];origins={};types={};gains=[];delta=[];action=[]
  for path in files:
   d=np.load(path);meta=json.loads(path.with_name(path.stem+'_meta.json').read_text());origin=json.loads(path.with_name(path.stem+'_origin.json').read_text());s=meta['scenario'];before=meta['before'];key=(s,before);coverage=origin['coverage'];origins[key]=dict(scenario=s,before=before,speed=coverage['speed'],curvature_proxy=abs(coverage.get('lane_heading_change_10m',0.))/10.,traffic_count=coverage['traffic_count'],lane_type=coverage['lane_type']);names=['baseline','gaussian','mean','best_elite'] if path.is_relative_to(prior) else ['B','gaussian_1','gaussian_2'];idx=names.index(path.stem);pool=np.asarray(origin['candidate_actions']);amp=float(np.sqrt(np.mean((pool[idx,:10]-pool[0,:10])**2)));action.append(amp);ret=float(d['rtg'][0]);gains.append(dict(scenario=s,before=before,branch=path.stem,return_value=ret,amp=amp));old=s<62000
   rows.append(dict(scenario=s,before=before,branch=path.stem,old=old,crash=float(meta['crash']),out_of_road=float(meta['out_of_road']),success=float(meta['success']),actual_return=ret,action_rmse=amp));types[path.stem]=types.get(path.stem,0)+1
  for key in origins:
   group=[x for x in gains if (x['scenario'],x['before'])==key]
   for i,left in enumerate(group):
    for right in group[i+1:]:delta.append(left['return_value']-right['return_value'])
  report[role]=dict(scenes=len(set(x['scenario'] for x in rows)),origins=len(origins),branches=len(rows),crash_count=int(sum(x['crash'] for x in rows)),oor_count=int(sum(x['out_of_road'] for x in rows)),success_count=int(sum(x['success'] for x in rows)),candidate_types=types,by_batch={age:dict(branches=sum(x['old']==(age=='old') for x in rows),crash=int(sum(x['crash'] for x in rows if x['old']==(age=='old'))),oor=int(sum(x['out_of_road'] for x in rows if x['old']==(age=='old')))) for age in ['old','new']},origin_features={k:describe([x[k] for x in origins.values()]) for k in ['speed','curvature_proxy','traffic_count']},road_types={name:sum(x['lane_type']==name for x in origins.values()) for name in sorted(set(x['lane_type'] for x in origins.values()))},action_perturbation=describe(action),actual_pair_delta=describe(delta),abs_pair_delta=describe(np.abs(delta)),**cluster_intervals(rows,{k:(k,None) for k in ['crash','out_of_road','success','actual_return','action_rmse']},samples=p['bootstrap_samples'],seed=p['bootstrap_seed']))
  all_origins[role]=list(origins.values());allrows.extend([dict(role=role,**x) for x in rows])
 training=all_origins['Fit']+all_origins['InnerStop'];strategy=dict(status='proposed only; NOT executed this round',source_maps='Fit+InnerStop ONLY; no seenValidation features/returns for thresholds',speed_cutpoints=np.quantile([x['speed'] for x in training],[1/3,2/3]).tolist(),curvature_proxy_cutpoints=np.quantile([x['curvature_proxy'] for x in training],[1/3,2/3]).tolist(),traffic_count_cutpoints=np.quantile([x['traffic_count'] for x in training],[1/3,2/3]).tolist(),road_types=sorted(set(x['lane_type'] for x in training)),scan_steps=[25,50,100,150,250],scene_pool=[70000,71999],seed=2026111704,selection='Visit a pre-shuffled fresh scenario pool with unchanged SAC; at fixed reachable scan steps classify current speed/absolute heading-change10m/traffic_count/lane_type. Fill under-represented observable strata to fixed equal quotas, ties by predetermined order. Generate same reward-blind candidate definitions. Record native unreachable origins; never replace/select using future Return/Crash/OOR. Split by scene BEFORE outcomes and use grouped CV; no dense extra windows.',limitation='traffic_count is an available traffic-object-count proxy, NOT vehicles/km; curvature is local reference heading change per10m, not exact complete map curvature.')
 fold_distribution=[]
 for i,heldout in enumerate(p['cv_folds']):
  sets={}
  for label,select in [('training',lambda s:s not in heldout),('heldout',lambda s:s in heldout)]:
   rr=[r for r in allrows if r['role']!='SeenValidation' and select(r['scenario'])];oo=[r for r in training if select(r['scenario'])]
   sets[label]=dict(scenes=len(set(r['scenario'] for r in rr)),branches=len(rr),crash_count=int(sum(r['crash'] for r in rr)),oor_count=int(sum(r['out_of_road'] for r in rr)),origin_features={key:describe([r[key] for r in oo]) for key in ['speed','curvature_proxy','traffic_count']},action_rmse=describe([r['action_rmse'] for r in rr]),actual_return=describe([r['actual_return'] for r in rr]))
  fold_distribution.append(dict(fold=i,heldout_maps=heldout,**sets))
 dump(R/'next_collection_strategy.json',strategy);dump(R/'distribution.json',dict(cohorts=report,cv_fold_distribution=fold_distribution,origin_rows=all_origins,branch_rows=allrows,correction='11/0 OOR counts belonged only to96/32 NEW branches. FULL cohorts144/53 contain23/10 OOR; InnerStop does contain native failures. Do not attribute overfit to an entirely failure-free53-branch set.',scope='Descriptive seen data only. CV is restricted to28 Fit+InnerStop maps; seen validation never selects a model, threshold, fold or next-sampling stratum. Outcomes correlated within origin/map.'))
 print({k:(x['branches'],x['oor_count'],x['crash_count']) for k,x in report.items()},flush=True)
if __name__=='__main__':main()
