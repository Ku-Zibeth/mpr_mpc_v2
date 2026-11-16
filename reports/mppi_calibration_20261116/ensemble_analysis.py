"""Distinguish the five-Q mean decision from average per-member correctness."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from evaluation_statistics import cluster_intervals
result={}
for split in ('dev','validation','diagnosis'):
 data=json.loads((R/(split+'_probe_analysis.json')).read_text());cases=data['cases'];rows=[]
 for row in cases:
  if row['branch']=='B' or row['branch']=='elite_Q1' or 'actual_SAC_RTG' not in row:continue
  b=next(x for x in cases if x['branch']=='B' and x['scenario']==row['scenario'] and x['before']==row['before'])
  if 'actual_SAC_RTG' not in b:continue
  delta=row['actual_SAC_RTG']-b['actual_SAC_RTG'];r=dict(scenario=row['scenario'],before=row['before'],configuration=row['branch'])
  for field,name in [('predicted_Q_mean','predicted_latent_accuracy'),('actual_z_candidate_Q_mean','real_latent_same_action_accuracy'),('actual_z_SAC_Q_mean','real_latent_sac_action_accuracy')]:
   pred=row[field]-b[field];r[name]=None if abs(delta)<=1e-6 else .5 if abs(pred)<=1e-6 else float(np.sign(pred)==np.sign(delta))
  rows.append(r)
 keys=('predicted_latent_accuracy','real_latent_same_action_accuracy','real_latent_sac_action_accuracy')
 summary=[dict(configuration=g,**cluster_intervals([r for r in rows if r['configuration']==g],{k:(k,None) for k in keys},samples=2000,seed=1)) for g in sorted({r['configuration'] for r in rows})]
 result[split]=dict(summary=summary,rows=rows)
(R/'ensemble_terminal_ranking.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result['validation']['summary'],indent=2))
