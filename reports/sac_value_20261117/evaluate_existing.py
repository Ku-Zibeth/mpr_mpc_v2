"""One fixed latent-only model; old matched pools are development/validation only."""
import json,sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'))
from analyze import correlation,pair_accuracy
from probe import dump
import numpy as np,torch
from evaluation_statistics import cluster_intervals
from planning.sac_value import SACContinuationValue,load_sac_value,state_sha256
from tools.value_calibration import assert_origin_match
p=json.loads((R/'protocol.json').read_text());main,artifact=load_sac_value(R/'models/sac_value_candidate.pt',wm_checkpoint=p['checkpoint'],horizon=10,gamma=.995,device='cpu',require_validated=False)
oracle_art=torch.load(R/'models/actual_latent_oracle.pt',map_location='cpu',weights_only=True);oracle=SACContinuationValue(**oracle_art['architecture']);oracle.load_state_dict(oracle_art['state_dict']);oracle.freeze()
results={}
with torch.no_grad():
 for split in ('dev','validation'):
  rows=[];predictions=[]
  for folder in sorted(Path(p['old_data']).glob(split+'_*')):
   ref=json.loads((folder/'B_origin.json').read_text());s=ref['episode_config']['scenario'];before=ref['episode_config']['before']
   names=['B','Q1','Q05','Q025','Q0']+(['elite_Q1'] if (folder/'elite_Q1.json').exists() else [])
   data=[json.loads((folder/(n+'.json')).read_text()) for n in names]
   for n in names:assert_origin_match(ref,json.loads((folder/(n+'_origin.json')).read_text()))
   assert all(x['gamma']==float(artifact['context']['gamma']) for x in data)
   z=torch.tensor([np.asarray(x['predicted_terminal_latent']).reshape(-1).tolist() for x in data])
   actual_z=torch.tensor([np.asarray(x['prefix'][-1]['actual_z_after']).reshape(-1).tolist() for x in data])
   vv=main(z).cpu().numpy();av=oracle(actual_z).cpu().numpy();qy=np.array([x['predicted_q'] for x in data]);y=np.array([x['actual_continuation'] for x in data])
   rp=np.array([x['predicted_prefix'] for x in data]);d=np.array([x['terminal_discount'] for x in data]);real=np.array([x['actual_discounted_return'] for x in data])
   for model,v in [('old_Q',qy),('predicted_V_SAC',vv),('actual_latent_oracle',av)]:
    j=rp+d*v;mae=float(np.mean(abs(j-real)))
    rows.append(dict(scenario=s,before=before,model=model,terminal_pair_accuracy=pair_accuracy(v,y),terminal_spearman=correlation(v,y,True),total_pair_accuracy=pair_accuracy(j,real),
      total_spearman=correlation(j,real,True),top1_regret=float(real.max()-real[j.argmax()]),v_mae=float(np.mean(abs(v-y))),j_mae=mae,j_bias=float(np.mean(j-real))))
    predictions.append(dict(scenario=s,before=before,model=model,labels=names,predicted_value=v.tolist(),real_continuation=y.tolist(),predicted_return=j.tolist(),real_return=real.tolist(),
      selected_label=names[int(j.argmax())],selected_real_return=float(real[j.argmax()]),baseline_real_return=float(real[0])))
  summary=[]
  keys=('terminal_pair_accuracy','terminal_spearman','total_pair_accuracy','total_spearman','top1_regret','v_mae','j_mae','j_bias')
  for model in ('old_Q','predicted_V_SAC','actual_latent_oracle'):
   rr=[x for x in rows if x['model']==model];summary.append(dict(model=model,origins=len(rr),scenes=len({x['scenario'] for x in rr}),**cluster_intervals(rr,{k:(k,None) for k in keys},samples=2000,seed=1)))
  results[split]=dict(summary=summary,rows=rows,predictions=predictions)
  print(split,summary,flush=True)
val={x['model']:x for x in results['validation']['summary']};v=val['predicted_V_SAC'];old=val['old_Q']
passed=bool(v['terminal_pair_accuracy_ci95_low'] is not None and v['terminal_pair_accuracy_ci95_low']>.5 and v['terminal_spearman_ci95_low'] is not None and v['terminal_spearman_ci95_low']>0 and v['j_mae']<old['j_mae'])
evidence=dict(passed=passed,scenarios=p['validation_scenarios'],pair_accuracy_ci95_low=v['terminal_pair_accuracy_ci95_low'],spearman_ci95_low=v['terminal_spearman_ci95_low'],
 j_mae=v['j_mae'],old_q_j_mae=old['j_mae'],head_state_sha256=state_sha256(artifact['state_dict']),
 interpretation='Independent predicted-latent ranking and J-error criteria passed; driving benefit still needs actual development/independent tests.' if passed else 'Predicted-latent ranking and/or error criteria failed. This candidate must not be configured for online use.')
if passed:
 artifact['validation']=evidence;path=R/'models/sac_value_rank_validated.pt';assert not path.exists();torch.save(artifact,path)
dump(R/'existing_pool_evaluation.json',dict(results=results,validation=evidence,
 scope='Original32 origins/172 branches remain untouched. H=10 prefixes then original SAC. No development/validation labels entered model fitting or epoch selection. Old Q(a_H) is a diagnostic baseline only; V_SAC targets state continuation, never the unexecuted a_H. Oracle actual-latent model is not deployable. CI resamples scenes, not candidate pairs.'))
