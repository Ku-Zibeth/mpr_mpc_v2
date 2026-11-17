"""Real fixed-origin branch outcomes and frozen-scorer pool selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from evaluation_statistics import cluster_intervals
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump


def main():
 p=json.loads((R/'protocol.json').read_text());ev=json.loads((R/'ranking_evaluation.json').read_text());rows=[];predictions={name:{(x['scenario'],x['before']):x for x in v['selections']} for name,v in ev['results']['validation'].items()};paired=[];prefix_error=[]
 for s in p['validation_scenarios']:
  for before in p['origin_steps']:
   folder=R/'data'/f'{s}_{before}'
   if not all((folder/(b+'.npz')).exists() for b in p['branches']):continue
   outcomes={b:json.loads((folder/(b+'_meta.json')).read_text()) for b in p['branches']};base=outcomes['baseline']
   for name in p['branches']:
    m=outcomes[name];d=np.load(folder/(name+'.npz'));gain=float(m['actual_return']-base['actual_return']);prefix=sum(p['gamma']**i*x for i,x in enumerate(d['rewards'][:10]));prefix_error.append(dict(scenario=s,before=before,branch=name,predicted_prefix=float(d['predicted_prefix']),actual_prefix=prefix,prefix_error=float(d['predicted_prefix'])-prefix,imagined_actual_terminal_l2=float(np.linalg.norm(d['predicted_terminal']-d['actual_terminal'])) if len(d['actual_terminal']) else None))
    rows.append(dict(scenario=s,before=before,method='candidate_'+name,success=float(m['success']),crash=float(m['crash']),out_of_road=float(m['out_of_road']),route_completion=m['route_completion'],episode_return=m['episode_return'],actual_discounted_return=m['actual_return'],average_safety_cost=m['cost_sum'],planning_time_ms=m['planning_time_ms'],predicted_gain=float(d['predicted_return'])-float(np.load(folder/'baseline.npz')['predicted_return']),actual_return_gain=gain))
   for name,origins in predictions.items():
    chosen=origins[(s,before)];m=outcomes[chosen['selected']];bidx=chosen['labels'].index('baseline');ix=chosen['labels'].index(chosen['selected']);gain=chosen['scores'][ix]-chosen['scores'][bidx]
    rows.append(dict(scenario=s,before=before,method='pool_selected_'+name,success=float(m['success']),crash=float(m['crash']),out_of_road=float(m['out_of_road']),route_completion=m['route_completion'],episode_return=m['episode_return'],actual_discounted_return=m['actual_return'],average_safety_cost=m['cost_sum'],planning_time_ms=m['planning_time_ms'],predicted_gain=gain,actual_return_gain=m['actual_return']-base['actual_return']))
    paired.append(dict(scenario=s,before=before,model=name,selected=chosen['selected'],predicted_gain=gain,actual_return_gain=m['actual_return']-base['actual_return'],crash_change=float(m['crash'])-float(base['crash']),oor_change=float(m['out_of_road'])-float(base['out_of_road'])))
 keys=['success','crash','out_of_road','route_completion','episode_return','actual_discounted_return','average_safety_cost','planning_time_ms','predicted_gain','actual_return_gain'];summaries={}
 for name in sorted(set(r['method'] for r in rows)):
  rr=[r for r in rows if r['method']==name];summaries[name]=dict(origins=len(rr),scenes=len(set(x['scenario'] for x in rr)),**cluster_intervals(rr,{k:(k,None) for k in keys},samples=p['bootstrap_samples'],seed=p['bootstrap_seed']))
 dump(R/'counterfactual_results.json',dict(summary=summaries,rows=rows,paired=paired,prefix_error=prefix_error,note='Real H10 interventions followed by frozen original SAC to native done. Candidate definitions are pre-reward original-Q Gaussian/Mean/Elite; selecting one from this already executed pool tests ranking, NOT new-score full sequential MPPI or full-episode closed-loop driving. No new score was enabled online. Suffix episode metrics start at origin; original-Q search times measured under shared GPU worker load are not a latency comparison.'))
if __name__=='__main__':main()
