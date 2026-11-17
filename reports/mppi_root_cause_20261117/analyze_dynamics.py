"""Paired relative latent errors; physical bicycle errors are explicitly separate."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from scipy.stats import spearmanr
from tools.mppi_diagnosis import relative_dynamics,angle_difference
from evaluation_statistics import cluster_intervals
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump


def summarize(rows,keys,p):
 return cluster_intervals(rows,{k:(k,None) for k in keys},samples=p['bootstrap_samples'],seed=p['bootstrap_seed'])


def main():
 p=json.loads((R/'protocol.json').read_text());rows=[];vectors_pred=[];vectors_real=[];vectors_physical=[]
 for folder in sorted((R/'states').iterdir()):
  paths=sorted(folder.glob('*.npz'));base=next((x for x in paths if x.stem in ['B','baseline']),None)
  if base is None:continue
  a0=np.load(base);baseline=json.loads(base.with_suffix('.json').read_text());base_source=np.load(baseline['source']);initial_yaw=a0['actual_physical'][0,2];normal=np.array([-np.sin(initial_yaw),np.cos(initial_yaw)])
  for path in paths:
   if path==base:continue
   a=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());source=np.load(meta['source']);gain=float(a['actual_return']-a0['actual_return']);amplitude=float(np.sqrt(np.mean((a['commands'][:10]-a0['commands'][:10])**2)))
   for hi,h in enumerate(p['horizons']):
    if min(len(a['actual']),len(a0['actual']))<=h:continue
    dp=relative_dynamics(np.stack([a0['predicted'][h],a['predicted'][h]]),np.stack([a0['actual'][h],a['actual'][h]]));realnorm=float(dp['actual_delta_l2'][1]);prednorm=float(dp['predicted_delta_l2'][1]);error=float(dp['error_l2'][1]);qp=a['q_pred_own'][hi].mean()-a0['q_pred_own'][hi].mean();qr=a['q_real_own'][hi].mean()-a0['q_real_own'][hi].mean();qpc=a['q_pred_common'][hi].mean()-a0['q_pred_common'][hi].mean();qrc=a['q_real_common'][hi].mean()-a0['q_real_common'][hi].mean();pairpred=a['q_pred_own'][hi,:2].mean()-a0['q_pred_own'][hi,:2].mean();pairreal=a['q_real_own'][hi,:2].mean()-a0['q_real_own'][hi,:2].mean()
    predlat=float((a['kinematic_xy'][h]-a0['kinematic_xy'][h])@normal);reallat=float((a['actual_physical'][h,:2]-a0['actual_physical'][h,:2])@normal);predyaw=angle_difference(a['kinematic_yaw'][h],a0['kinematic_yaw'][h]);realyaw=angle_difference(a['actual_physical'][h,2],a0['actual_physical'][h,2]);predspeed=a['kinematic_speed'][h]-a0['kinematic_speed'][h];realspeed=a['actual_physical'][h,3]-a0['actual_physical'][h,3]
    vectors_pred.append(dp['predicted_delta'][1]);vectors_real.append(dp['actual_delta'][1]);vectors_physical.append([reallat,float(realyaw),float(realspeed)])
    prefix_delta=sum(p['gamma']**j*(source['rewards'][j]-base_source['rewards'][j]) for j in range(h));predj=prefix_delta+p['gamma']**h*qp;realj=prefix_delta+p['gamma']**h*qr;effective=abs(gain)>1.
    predcorrect=bool(predj*gain>0);realcorrect=bool(realj*gain>0)
    rows.append(dict(scenario=meta['scenario'],before=meta['before'],branch=path.stem,h=h,relative_error_l2=error,cosine=float(dp['cosine'][1]),actual_delta_l2=realnorm,predicted_delta_l2=prednorm,relative_error_ratio=error/realnorm if realnorm>1e-6 else None,wrong_direction=float(dp['cosine'][1]<0) if np.isfinite(dp['cosine'][1]) else None,return_gain=gain,oracle_prefix_predicted_z_q_accuracy=float(predcorrect) if effective else None,oracle_prefix_real_z_q_accuracy=float(realcorrect) if effective else None,real_z_accuracy_improvement=float(realcorrect)-float(predcorrect) if effective else None,latent_replacement_repairs=float(not predcorrect and realcorrect) if effective else None,latent_replacement_harms=float(predcorrect and not realcorrect) if effective else None,absolute_return_gain=abs(gain),action_rmse=amplitude,
     q_flip_ensemble=float(qp*qr<0) if min(abs(qp),abs(qr))>1e-5 else None,q_flip_common_action=float(qpc*qrc<0) if min(abs(qpc),abs(qrc))>1e-5 else None,q_flip_fixed_pair01=float(pairpred*pairreal<0) if min(abs(pairpred),abs(pairreal))>1e-5 else None,
     kinematic_relative_lateral_error_m=abs(predlat-reallat),kinematic_relative_yaw_error_rad=float(abs(angle_difference(predyaw,realyaw))),kinematic_relative_speed_error_mps=abs(predspeed-realspeed),actual_relative_lateral_m=reallat,predicted_relative_lateral_m=predlat))
 keys=['relative_error_l2','cosine','actual_delta_l2','predicted_delta_l2','relative_error_ratio','wrong_direction','q_flip_ensemble','q_flip_common_action','q_flip_fixed_pair01','oracle_prefix_predicted_z_q_accuracy','oracle_prefix_real_z_q_accuracy','real_z_accuracy_improvement','latent_replacement_repairs','latent_replacement_harms','kinematic_relative_lateral_error_m','kinematic_relative_yaw_error_rad','kinematic_relative_speed_error_mps'];byh={};relationships={};bands={}
 for h in p['horizons']:
  rr=[x for x in rows if x['h']==h];byh[str(h)]=dict(pairs=len(rr),scenes=len(set(x['scenario'] for x in rr)),**summarize(rr,keys,p));rel=[]
  for s in sorted(set(x['scenario'] for x in rr)):
   r=[x for x in rr if x['scenario']==s];x=[z['relative_error_l2'] for z in r];y=[z['absolute_return_gain'] for z in r]
   rel.append(dict(scenario=s,error_vs_abs_return_spearman=float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else None))
  relationships[str(h)]=summarize(rel,['error_vs_abs_return_spearman'],p)
  b=[]
  for lo,hi in [(0,.02),(.02,.05),(.05,.1),(.1,np.inf)]:
   ss=[x for x in rr if lo<=x['action_rmse']<hi];b.append(dict(low=lo,high=float(hi) if np.isfinite(hi) else None,pairs=len(ss),**(summarize(ss,keys,p) if ss else {})))
  bands[str(h)]=b
 np.savez_compressed(R/'relative_vectors.npz',predicted=np.stack(vectors_pred),actual=np.stack(vectors_real),physical=np.asarray(vectors_physical),scenario=np.array([r['scenario'] for r in rows]),before=np.array([r['before'] for r in rows]),h=np.array([r['h'] for r in rows]))
 dump(R/'relative_dynamics.json',dict(by_horizon=byh,error_return_relationship=relationships,action_bands=bands,rows=rows,scope='Fit+InnerStop only; same origin candidate-minus-baseline, not absolute latent distance. Cosine undefined for zero change. All members share the same frozen Q; ensemble mean and pair01 are diagnostic readouts, not online aggregation changes. Q accuracy comparison uses the SAME REAL prefix reward on both sides, changing ONLY predicted-vs-real latent before the same frozen Q/terminal action. This is an offline oracle decomposition, not an executable score; h<10 continuation still includes the remaining fixed prefix. Physical errors are from EXISTING KINEMATIC bicycle model; WM has no physical decoder. Lateral is initial ego-body-frame displacement, never latent meters.'))
 print(byh,flush=True)
if __name__=='__main__':main()
