"""Matched real512/real535/imagined512 group-CV linear probes, no online head."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from scipy.stats import spearmanr,kendalltau
from tools.mppi_diagnosis import effective_df_ridge
from evaluation_statistics import cluster_intervals
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump


def metrics(scores,items,p):
 rows=[]
 for key in sorted({(x['scenario'],x['before']) for x in items}):
  ids=[i for i,x in enumerate(items) if (x['scenario'],x['before'])==key];ss=scores[ids];y=np.array([items[i]['continuation'] for i in ids]);pairs=[(i,j) for i in range(len(ids)) for j in range(i+1,len(ids))];effective=[(i,j) for i,j in pairs if abs(y[i]-y[j])>1.];err=[abs((ss[i]-ss[j])-(y[i]-y[j])) for i,j in pairs];actualprefix=np.array([items[i]['prefix'] for i in ids]);realj=actualprefix+p['gamma']**10*y;predj=actualprefix+p['gamma']**10*ss
  rows.append(dict(scenario=key[0],before=key[1],candidates=len(ids),pairs=len(pairs),effective_pairs=len(effective),accuracy=float(np.mean([(ss[i]-ss[j])*(y[i]-y[j])>0 for i,j in effective])) if effective else None,spearman=float(spearmanr(ss,y).statistic) if np.std(ss)>0 and np.std(y)>0 else None,kendall=float(kendalltau(ss,y).statistic) if np.std(ss)>0 and np.std(y)>0 else None,regret=float(y.max()-y[ss.argmax()]),difference_mae=float(np.mean(err)),total_j_regret=float(realj.max()-realj[predj.argmax()]),predicted=ss.tolist(),actual=y.tolist(),branches=[items[i]['branch'] for i in ids]))
 summary=cluster_intervals(rows,{k:(k,None) for k in ['accuracy','spearman','kendall','regret','difference_mae','total_j_regret']},samples=p['bootstrap_samples'],seed=p['bootstrap_seed']);return dict(summary=summary,rows=rows)


def main():
 p=json.loads((R/'protocol.json').read_text());items=[];excluded=[];arrays=[];metadata=[]
 for folder in sorted((R/'states').iterdir()):
  for path in sorted(folder.glob('*.npz')):
   meta=json.loads(path.with_suffix('.json').read_text());assert meta['scenario'] in p['cv_scenarios'];data=np.load(path)
   if len(data['state535'])!=535:excluded.append(meta);continue
   assert np.array_equal(data['state535'][:512],data['actual'][10])
   items.append(dict(scenario=meta['scenario'],before=meta['before'],branch=meta['branch'],continuation=float(data['actual_continuation']),prefix=float(data['actual_prefix']),source=str(path)));arrays.append((data['state535'],data['predicted'][10]));metadata.append(meta)
 assert len(items)+len(excluded)==197
 full=np.stack([x[0] for x in arrays]);imagined=np.stack([x[1] for x in arrays]);y=np.array([x['continuation'] for x in items]);scene=np.array([x['scenario'] for x in items]);cases={}
 groups=dict(path=(512,524),wm_return=(524,525),feasibility=(525,527),risk=(527,535))
 features=dict(real512=full[:,:512],real535=full,imagined512=imagined)
 for name,(lo,hi) in groups.items():features['real535_without_'+name]=np.concatenate([full[:,:lo],full[:,hi:]],1)
 outputs={name:np.full(len(items),np.nan) for name in features};folds=[]
 for fold,heldout in enumerate(p['cv_folds']):
  train=np.flatnonzero(~np.isin(scene,heldout));test=np.flatnonzero(np.isin(scene,heldout));assert set(scene[train]).isdisjoint(scene[test]);pairs=[];weights=[];samplew=np.zeros(len(train));localgroups={key:[j for j,i in enumerate(train) if (items[i]['scenario'],items[i]['before'])==key] for key in {(items[i]['scenario'],items[i]['before']) for i in train}};maps={s:sum(k[0]==s for k in localgroups) for s in set(scene[train])}
  for key,ids in localgroups.items():
   factor=1/maps[key[0]];local=[(i,j) for a,i in enumerate(ids) for j in ids[a+1:]];pairs.extend(local);weights.extend([factor/len(local)]*len(local))
   for i in ids:samplew[i]=factor/len(ids)
  pairs=np.array(pairs);samplew/=samplew.sum();fitlabels=y[train][pairs[:,0]]-y[train][pairs[:,1]];stats={}
  for name,x in features.items():
   xt=x[train];mu=(samplew[:,None]*xt).sum(0);sd=np.sqrt((samplew[:,None]*(xt-mu)**2).sum(0)).clip(.001);normalized=(xt-mu)/sd;dx=normalized[pairs[:,0]]-normalized[pairs[:,1]];coef,info=effective_df_ridge(dx,fitlabels,weights,degrees=p['probe']['effective_df']);outputs[name][test]=((x[test]-mu)/sd)@coef;stats[name]=info
  folds.append(dict(fold=fold,training_maps=sorted(set(scene[train].tolist())),heldout_maps=heldout,test_candidates=len(test),train_pairs=len(pairs),probe_stats=stats))
 assert all(np.isfinite(x).all() for x in outputs.values());results={name:metrics(score,items,p) for name,score in outputs.items()};comparisons={}
 for reference in ['real512','imagined512',*['real535_without_'+name for name in groups]]:
  a={(r['scenario'],r['before']):r for r in results[reference]['rows']};rr=[]
  for row in results['real535']['rows']:
   other=a[(row['scenario'],row['before'])];rr.append(dict(scenario=row['scenario'],accuracy_gain=None if row['accuracy'] is None or other['accuracy'] is None else row['accuracy']-other['accuracy'],regret_improvement=other['regret']-row['regret']))
  comparisons[reference]=cluster_intervals(rr,{k:(k,None) for k in ['accuracy_gain','regret_improvement']},samples=p['bootstrap_samples'],seed=p['bootstrap_seed'])
 actor={}
 for i,name in enumerate(groups):
  effects=np.stack([np.load(x['source'])['actor_effects'][i] for x in items]);norm=np.linalg.norm(effects,axis=1);actor[name]=dict(median_action_change=float(np.median(norm)),p95_action_change=float(np.quantile(norm,.95)),max_action_change=float(norm.max()),fraction_over_001=float(np.mean(norm>.01)))
 varying=[]
 for j in range(512,535):varying.append(dict(index=j,mean=float(full[:,j].mean()),std=float(full[:,j].std()),min=float(full[:,j].min()),max=float(full[:,j].max())))
 dump(R/'state_information.json',dict(candidates=len(items),excluded_native_ended=excluded,scenes=len(set(scene)),origins=len(set((x['scenario'],x['before']) for x in items)),folds=folds,results=results,paired_full535_comparisons=comparisons,actor_zero_block_sensitivity=actor,extra_feature_distribution=varying,risk_feature_order=metadata[0]['risk_feature_order'],
  interpretation='Only Fit+InnerStop maps; 5 folds fixed before outcomes. Real H-state535 is produced by original builder. Same MC continuation labels and matched origins across all inputs. Fit-fold-only standardization and label-free effective_df16 ridge. Coefficients ephemeral; no sixth value artifact, no online input change. Failure of these linear probes cannot prove information-theoretic insufficiency. Actor zero-block sensitivity is not a causal driving-benefit experiment.'))
 print({n:r['summary'] for n,r in results.items()},flush=True)
if __name__=='__main__':main()
