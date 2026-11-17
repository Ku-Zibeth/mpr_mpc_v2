"""Fixed low-capacity physical interpretation; no persistent decoder or value head."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from tools.mppi_diagnosis import effective_df_ridge
from evaluation_statistics import cluster_intervals
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump


def main():
 p=json.loads((R/'protocol.json').read_text());a=np.load(R/'relative_vectors.npz');xreal=a['actual'];xpred=a['predicted'];target=a['physical'];scenes=a['scenario'];hs=a['h'];before=a['before'];predreal=np.full_like(target,np.nan);predimag=np.full_like(target,np.nan);folds=[]
 for h in p['horizons']:
  for fold,heldout in enumerate(p['cv_folds']):
   train=np.flatnonzero((hs==h)&~np.isin(scenes,heldout));test=np.flatnonzero((hs==h)&np.isin(scenes,heldout));weights=np.zeros(len(train));groups={key:np.flatnonzero((scenes[train]==key[0])&(before[train]==key[1])) for key in set(zip(scenes[train],before[train]))};counts={s:sum(key[0]==s for key in groups) for s in set(scenes[train])}
   for key,ii in groups.items():weights[ii]=1./counts[key[0]]/len(ii)
   weights/=weights.sum();std=np.sqrt((weights[:,None]*xreal[train]**2).sum(0)).clip(.001);xt=xreal[train]/std
   stats=[]
   for j in range(3):
    coef,info=effective_df_ridge(xt,target[train,j],weights,degrees=16);predreal[test,j]=(xreal[test]/std)@coef;predimag[test,j]=(xpred[test]/std)@coef;stats.append(info)
   folds.append(dict(h=h,fold=fold,heldout_maps=heldout,training_pairs=len(train),test_pairs=len(test),capacity=stats))
 assert np.isfinite(predreal).all() and np.isfinite(predimag).all();rows=[]
 for i in range(len(target)):
  row=dict(scenario=int(scenes[i]),before=int(before[i]),h=int(hs[i]))
  for j,name in enumerate(['lateral_m','yaw_rad','speed_mps']):
   real=abs(predreal[i,j]-target[i,j]);imag=abs(predimag[i,j]-target[i,j]);row.update({name+'_real_latent_readout_error':float(real),name+'_imagined_latent_readout_error':float(imag),name+'_excess_error':float(imag-real),name+'_model_induced_readout_change':float(abs(predimag[i,j]-predreal[i,j]))})
  rows.append(row)
 keys=[k for k in rows[0] if k not in ['scenario','before','h']];byh={str(h):cluster_intervals([r for r in rows if r['h']==h],{k:(k,None) for k in keys},samples=p['bootstrap_samples'],seed=p['bootstrap_seed']) for h in p['horizons']}
 dump(R/'physical_readout.json',dict(by_horizon=byh,rows=rows,folds=folds,scope='Ephemeral5-fold linear readouts, effective_df16 from trainingX only; Fit+Inner maps only. Fit actual relative latent to true physical candidate-minus-baseline changes. Apply IDENTICAL readout to real vs imagined heldout delta. Real-latent error is the decoder floor; excess is controlled diagnostic evidence, NOT a direct calibrated WM physical-state output. No neural decoder/weight file, no online state changes.'))
 print(byh,flush=True)
if __name__=='__main__':main()
