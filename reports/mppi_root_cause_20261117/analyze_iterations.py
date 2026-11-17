"""Actual native branch returns for fixed per-round elites/means and intervention A/B."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from scipy.stats import spearmanr,kendalltau
from evaluation_statistics import cluster_intervals
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump


def summary(rows,keys,p):return cluster_intervals(rows,{k:(k,None) for k in keys},samples=p['bootstrap_samples'],seed=p['bootstrap_seed'])


def main():
 p=json.loads((R/'protocol.json').read_text());rows=[];iterations=[];agreements=[];slopes=[]
 for scenario,before in p['development_origins']:
  folder=R/'interventions'/f'{scenario}_{before}';catalog=np.load(R/'catalog'/f'{scenario}_{before}'/'catalog.npz');meta=json.loads((R/'catalog'/f'{scenario}_{before}'/'catalog.json').read_text());names=p['choices'];values=catalog['predicted_values'];byprotocol={}
  for protocol in ['A','B']:
   outcomes={n:json.loads((folder/f'{protocol}_{n}.json').read_text()) for n in names};byprotocol[protocol]=outcomes;base=outcomes['baseline']['actual_return'];basepred=float(values[0])
   for label in names:
    o=outcomes[label];idx=names.index(label);kind='baseline' if label=='baseline' else 'gaussian' if label=='gaussian' else 'mean' if label.startswith('mean') else 'elite';k=int(label[-1]) if kind in ['mean','elite'] else 0;delta=catalog['actions'][idx,:10]-catalog['actions'][0,:10]
    rows.append(dict(scenario=scenario,before=before,protocol=protocol,choice=label,kind=kind,k=k,predicted_gain=float(values[idx]-basepred),actual_gain=o['actual_return']-base,actual_return=o['actual_return'],episode_return=o['episode_return'],crash=float(o['crash']),out_of_road=float(o['out_of_road']),success=float(o['success']),route_completion=o['route_completion'],steer_rmse=float(np.sqrt(np.mean(delta[:,0]**2))),throttle_rmse=float(np.sqrt(np.mean(delta[:,1]**2))),first_steer_delta=float(delta[0,0]),first_throttle_delta=float(delta[0,1]),raw_elite_pred_gain=meta['raw_rounds'][k-1]['raw_best_gain'] if kind=='elite' else None,raw_elite_vs_baseline_accuracy=float(meta['raw_rounds'][k-1]['raw_best_gain']*(o['actual_return']-base)>0) if kind=='elite' and abs(o['actual_return']-base)>1. else None,elite_vs_baseline_accuracy=float((values[idx]-basepred)*(o['actual_return']-base)>0) if kind=='elite' and abs(o['actual_return']-base)>1. else None))
   for k in p['iterations_ablation']:
    selection=meta['iterations'][str(k)];o=outcomes[selection['selected']];iterations.append(dict(scenario=scenario,before=before,protocol=protocol,k=k,selected=selection['selected'],actual_gain=o['actual_return']-base,actual_return=o['actual_return'],predicted_gain=selection['predicted_gain'],crash=float(o['crash']),out_of_road=float(o['out_of_road']),success=float(o['success']),route_completion=o['route_completion']))
   for kind in ['mean','elite']:
    rr=[x for x in rows if x['scenario']==scenario and x['before']==before and x['protocol']==protocol and (x['kind']==kind or x['kind']=='baseline')];rr=sorted(rr,key=lambda x:x['k']);x=np.arange(5);yp=np.array([r['predicted_gain'] for r in rr]);ya=np.array([r['actual_gain'] for r in rr]);slopes.append(dict(scenario=scenario,protocol=protocol,kind=kind,predicted_slope=float(np.polyfit(x,yp,1)[0]),actual_slope=float(np.polyfit(x,ya,1)[0]),prediction_increases=float(np.all(np.diff(yp)>=0)),actual_decreases=float(np.all(np.diff(ya)<=0)),gain_4_minus_1=ya[-1]-ya[1],predicted_4_minus_1=yp[-1]-yp[1]))
  ya=np.array([byprotocol['A'][n]['actual_return'] for n in names]);yb=np.array([byprotocol['B'][n]['actual_return'] for n in names]);effective=[(i,j) for i in range(len(names)) for j in range(i+1,len(names)) if min(abs(ya[i]-ya[j]),abs(yb[i]-yb[j]))>1.]
  agreements.append(dict(scenario=scenario,before=before,pairs=len(effective),spearman=float(spearmanr(ya,yb).statistic),kendall=float(kendalltau(ya,yb).statistic),ranking_agreement=float(np.mean([(ya[i]-ya[j])*(yb[i]-yb[j])>0 for i,j in effective])) if effective else None,top1_same=float(ya.argmax()==yb.argmax()),baseline_return_A=float(ya[0]),baseline_return_B=float(yb[0]),A_returns=ya.tolist(),B_returns=yb.tolist(),labels=names))
 keys=['predicted_gain','actual_gain','actual_return','episode_return','crash','out_of_road','success','route_completion','steer_rmse','throttle_rmse','elite_vs_baseline_accuracy','raw_elite_pred_gain','raw_elite_vs_baseline_accuracy'];curve={};isummary={};ssummary={};pairdiffs=[]
 for protocol in ['A','B']:
  curve[protocol]={kind:{str(k):summary([r for r in rows if r['protocol']==protocol and ((k==0 and r['kind']=='baseline') or (r['kind']==kind and r['k']==k))],keys,p) for k in range(5)} for kind in ['mean','elite']}
  isummary[protocol]={str(k):summary([r for r in iterations if r['protocol']==protocol and r['k']==k],['actual_gain','actual_return','predicted_gain','crash','out_of_road','success','route_completion'],p) for k in p['iterations_ablation']}
  ssummary[protocol]={kind:summary([r for r in slopes if r['protocol']==protocol and r['kind']==kind],['predicted_slope','actual_slope','prediction_increases','actual_decreases','gain_4_minus_1','predicted_4_minus_1'],p) for kind in ['mean','elite']}
  for ka,kb in [(1,2),(1,4),(2,4)]:
   rr=[]
   for scenario,before in p['development_origins']:
    a=next(x for x in iterations if x['scenario']==scenario and x['before']==before and x['protocol']==protocol and x['k']==ka);b=next(x for x in iterations if x['scenario']==scenario and x['before']==before and x['protocol']==protocol and x['k']==kb);rr.append(dict(scenario=scenario,return_difference=b['actual_return']-a['actual_return'],crash_difference=b['crash']-a['crash'],oor_difference=b['out_of_road']-a['out_of_road']))
   pairdiffs.append(dict(protocol=protocol,reference=ka,candidate=kb,**summary(rr,['return_difference','crash_difference','oor_difference'],p)))
 # Only executed per-round top1/means/baseline have labels; not the64/Top8 pool.
 dump(R/'iteration_results.json',dict(curves=curve,iterations=isummary,paired_iterations=pairdiffs,slopes=ssummary,agreement=summary(agreements,['spearman','kendall','ranking_agreement','top1_same'],p),agreement_rows=agreements,rows=rows,iteration_rows=iterations,
  scope='Four already-seen Fit diagnostic maps, not independent generalization. A executesH10 thenSAC; B executesONE action thenSAC, NOT receding MPPI. Predicted curves share captured Q pair, original raw-round pair/gains retained in catalogs. Elite ranking accuracy is executed top1-versus-baseline only; no unobserved64/Top8 real ranking claim.1/2/4 standard planner outputs mapped to identical measured Mean/Baseline prefixes; only iterations changed.'))
 print('agreement',summary(agreements,['spearman','ranking_agreement'],p));print('slopes',ssummary);print('iterations',isummary,flush=True)
if __name__=='__main__':main()
