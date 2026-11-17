"""Single fresh-map validation after ALL heads/epochs are frozen; no tuning."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from scipy.stats import spearmanr,kendalltau
from dataset import build
from fit import scores,to_device
from planning.candidate_value import CandidateValue
from planning.sac_value import SACContinuationValue,state_sha256
from evaluation_statistics import cluster_intervals
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump


def summary(rows,p,keys):
 return dict(origins=len(rows),scenes=len({x['scenario'] for x in rows}),**cluster_intervals(rows,{k:(k,None) for k in keys},samples=p['bootstrap_samples'],seed=p['bootstrap_seed']))


def statistics(name,pred,d,p):
 rows=[];pairs=[];selections=[];tol=p['pair_tolerance'];truth=d['target'].cpu().numpy();pred=np.asarray(pred)
 for group in d['groups']:
  if not np.isfinite(pred[group]).all():continue
  first=d['items'][group[0]];scene,before=first['scene'],first['before'];yy=truth[group];ss=pred[group];base=next(j for j in group if d['items'][j]['branch'] in ('B','baseline'));mean=next((j for j in group if d['items'][j]['branch']=='mean'),None);select=group[int(np.argmax(ss))]
  pp=[(i,j) for a,i in enumerate(group) for j in group[a+1:]];effective=[(i,j) for i,j in pp if abs(truth[i]-truth[j])>tol]
  accuracy=float(np.mean([(pred[i]-pred[j])*(truth[i]-truth[j])>0 for i,j in effective])) if effective else None
  has_mean=mean is not None and abs(truth[mean]-truth[base])>tol
  higher=[j for j in group if j!=base and pred[j]-pred[base]>tol];bad=sum(truth[j]-truth[base]<-tol for j in higher)
  row=dict(scenario=scene,before=before,model=name,pair_accuracy=accuracy,all_pair_accuracy=float(np.mean([(pred[i]-pred[j])*(truth[i]-truth[j])>0 for i,j in pp])) if pp else None,
   baseline_mean_accuracy=float((pred[mean]-pred[base])*(truth[mean]-truth[base])>0) if has_mean else None,
   spearman=float(spearmanr(ss,yy).statistic) if np.std(ss)>0 and np.std(yy)>0 else None,kendall=float(kendalltau(ss,yy).statistic) if np.std(ss)>0 and np.std(yy)>0 else None,
   top1_regret=float(yy.max()-truth[select]),difference_mae=float(np.mean([abs((pred[i]-pred[j])-(truth[i]-truth[j])) for i,j in pp])) if pp else None,
   actual_return_gain=float(truth[select]-truth[base]),negative_selection=float(truth[select]-truth[base]<-tol),high_pred_bad_numerator=bad,high_pred_count=len(higher),high_prediction_drop_rate=bad/len(higher) if higher else None,
   predicted_gain=float(pred[select]-pred[base]),effective_pairs=len(effective),all_pairs=len(pp),selected=d['items'][select]['branch'])
  folder=(Path(p['old_data'])/'training_data' if first['source']=='old' else R/'data')/f'{scene}_{before}'
  coverage=json.loads((folder/('B_origin.json' if first['source']=='old' else 'baseline_origin.json')).read_text())['coverage']
  row.update(lane_type=coverage['lane_type'],curve_class='curved' if abs(coverage.get('lane_heading_change_10m',0.))>.05 else 'near_straight',traffic_class='0-4' if coverage['traffic_count']<5 else '5-9' if coverage['traffic_count']<10 else '10+',origin_speed=coverage['speed'])
  rows.append(row)
  for i,j in pp:
   real=float(truth[i]-truth[j]);delta=float(pred[i]-pred[j]);action=float(torch.mean((d['actions'][i,:10]-d['actions'][j,:10])**2).sqrt());pairs.append(dict(scenario=scene,before=before,model=name,left=d['items'][i]['branch'],right=d['items'][j]['branch'],real_delta=real,predicted_delta=delta,effective=abs(real)>tol,correct=float(delta*real>0),absolute_error=abs(delta-real),action_rmse=action))
  selections.append(dict(scenario=scene,before=before,model=name,labels=[d['items'][j]['branch'] for j in group],scores=ss.tolist(),real_returns=yy.tolist(),selected=d['items'][select]['branch'],baseline=d['items'][base]['branch']))
 keys=['pair_accuracy','all_pair_accuracy','baseline_mean_accuracy','spearman','kendall','top1_regret','difference_mae','actual_return_gain','negative_selection','predicted_gain']
 result=summary(rows,p,keys);result.update(high_prediction_drop=cluster_intervals(rows,{'rate':('high_pred_bad_numerator','high_pred_count')},samples=p['bootstrap_samples'],seed=p['bootstrap_seed']),effective_pairs=sum(x['effective_pairs'] for x in rows),all_pairs=sum(x['all_pairs'] for x in rows))
 stratified={}
 for axis,bounds in [('real_delta',p['difference_bands']),('action_rmse',p['action_rmse_bands'])]:
  groups=[]
  for lo,hi in zip(bounds[:-1],bounds[1:]):
   rr=[x for x in pairs if lo<=abs(x[axis])<(hi if hi is not None else np.inf)];active=[x for x in rr if x['effective']]
   # Within each map, average pairs before resampling whole maps.
   cluster=[dict(scenario=s,correct=float(np.mean([x['correct'] for x in active if x['scenario']==s])),absolute_error=float(np.mean([x['absolute_error'] for x in active if x['scenario']==s]))) for s in sorted({x['scenario'] for x in active})]
   groups.append(dict(low=lo,high=hi,pairs=len(rr),effective_pairs=len(active),**(summary(cluster,p,['correct','absolute_error']) if cluster else {})))
  stratified[axis]=groups
 for axis in ['lane_type','curve_class','traffic_class']:
  stratified[axis]={label:summary([r for r in rows if r[axis]==label],p,keys) for label in sorted({r[axis] for r in rows})}
 return dict(summary=result,rows=rows,pairs=pairs,selections=selections,stratified=stratified)


@torch.no_grad()
def main():
 p=json.loads((R/'protocol.json').read_text());manifest=json.loads((R/'dataset_manifest.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
 models={};hashes={}
 for name in p['models']:
  a=torch.load(R/'models'/f'{name}.pt',map_location=device,weights_only=True);head=SACContinuationValue(**a['architecture']) if name.startswith('absolute') else CandidateValue(**a['architecture']);head.to(device).load_state_dict(a['state_dict']);head.eval().requires_grad_(False);models[name]=head;hashes[name]=state_sha256(a['state_dict']);assert not set(a['training_scenarios'])&set(p['validation_scenarios'])
 old=torch.load(Path(p['old_data'])/'models/sac_value_candidate.pt',map_location=device,weights_only=True);head=SACContinuationValue(**old['architecture']).to(device);head.load_state_dict(old['state_dict']);head.freeze();models['previous_V_SAC']=head
 result={}
 for split in ['fit','inner_stop','validation']:
  d=to_device(build(split),device);predictions={}
  for name,head in models.items():
   if name=='previous_V_SAC':prediction=d['prefix']+d['discount']*head(d['terminal'])
   else:prediction,_=scores(head,name,d)
   predictions[name]=prediction.cpu().numpy()
  predictions['original_Q']=np.array([i['old_q'] for i in d['items']])
  results={name:statistics(name,pred,d,p) for name,pred in predictions.items()}
  # Paired per-origin regret changes; positive means the new model is better.
  for name in p['models']:
   rr={}
   for reference in ['original_Q','previous_V_SAC','rank_terminal']:
    oldrows={(x['scenario'],x['before']):x for x in results[reference]['rows']};rows=[dict(scenario=x['scenario'],regret_improvement=oldrows[(x['scenario'],x['before'])]['top1_regret']-x['top1_regret']) for x in results[name]['rows'] if (x['scenario'],x['before']) in oldrows]
    rr[reference]=summary(rows,p,['regret_improvement']) if rows else None
   results[name]['paired_regret']=rr
  result[split]=results;print(split,{name:(r['summary']['pair_accuracy'],r['summary']['top1_regret']) for name,r in results.items()},flush=True)
 validation=result['validation'];primary=validation[p['training']['primary']];v=primary['summary'];q=validation['original_Q']['summary'];oldv=validation['previous_V_SAC']['summary'];gain=primary['paired_regret']['original_Q']
 def gt(x,y):return x is not None and y is not None and x>y
 passed=bool(v['scenes']>=6 and gt(v['pair_accuracy_ci95_low'],.5) and gt(v['spearman_ci95_low'],0.) and gt(gain['regret_improvement_ci95_low'],0.) and gt(primary['paired_regret']['previous_V_SAC']['regret_improvement_ci95_low'],0.) and v['pair_accuracy']>max(q['pair_accuracy'],oldv['pair_accuracy']) and v['baseline_mean_accuracy'] is not None and v['baseline_mean_accuracy']>=q['baseline_mean_accuracy'] and v['negative_selection']<=q['negative_selection'])
 evidence=dict(passed=passed,primary=p['training']['primary'],scenarios=p['validation_scenarios'],primary_head_state_sha256=hashes[p['training']['primary']],criteria=p['acceptance'],
  interpretation='Fresh-map ranking qualified; separate development and untouched closed-loop evidence still required before default deployment.' if passed else 'Fresh-map ranking/regret requirements failed; keep all new heads offline. No larger closed-loop or100-scene test.')
 output=R/'ranking_evaluation.json';assert not output.exists();dump(output,dict(results=result,validation=evidence,head_hashes=hashes,scope='One fixed protocol, five frozen heads, a single new independent map set. No score-sign reversal, weight sweep or retraining after validation. Absolute ranking scores have no calibrated origin offset: report difference errors, not their meaningless absolute-return MAE.'))
if __name__=='__main__':main()
