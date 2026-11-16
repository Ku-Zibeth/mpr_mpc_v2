"""Decompose error without conflating Q training policy with realized SAC RTG."""
from pathlib import Path
import json,numpy as np
REPORT=Path(__file__).parent;protocol=json.loads((REPORT/'probe_protocol.json').read_text());rows=[];unavailable=[]
for scenario,before in protocol['origins']:
 try:b=json.loads((REPORT/f'probe_{scenario}_sac.json').read_text());d=json.loads((REPORT/f'probe_{scenario}_mean.json').read_text())
 except FileNotFoundError:unavailable.append(dict(scenario=scenario,reason='Missing raw branch, see worker log'));continue
 assert b['observed_state_id']==d['observed_state_id'],scenario
 for key in ('observation','baseline','mean','candidate_actions','common_pool_scores','prefix_controls'):assert b[key]==d[key],(scenario,key)
 assert b['episode_complete'] and d['episode_complete']
 i=b['variant_indices']['D0']['mean'];gamma=b['gamma'];H=len(b['baseline'])-1;gain=d['actual_discounted_return']-b['actual_discounted_return']
 r=dict(scenario=scenario,before=before,prefix_source=b['prefix_source'],matched_observed_state_and_candidates=True,
  actual_discounted_gain=gain,actual_prefix_gain=d['actual_first_H_discounted_reward']-b['actual_first_H_discounted_reward'],
  actual_discounted_continuation_gain=gamma**H*(d['actual_terminal_continuation']-b['actual_terminal_continuation']),
  baseline_success=b['success'],mean_success=d['success'],baseline_crash=b['crash'],mean_crash=d['crash'],baseline_offroad=b['offroad'],mean_offroad=d['offroad'],
  baseline_terminal_latent_rmse=b['latent_rollout_rmse'][-1] if len(b['latent_rollout_rmse'])==H else None,
  mean_terminal_latent_rmse=d['latent_rollout_rmse'][-1] if len(d['latent_rollout_rmse'])==H else None,
  actual_baseline_SAC_rtg=b['actual_terminal_continuation'],actual_mean_SAC_rtg=d['actual_terminal_continuation'],definitions={})
 for mode in ('D0','D2'):
  s=b['common_pool_scores'][mode];values=s['values'];rew=s['prefix_reward'];pq=np.asarray(s['q_members'])[:,i]-np.asarray(s['q_members'])[:,0]
  decomp=dict(predicted_gain=values[i]-values[0],predicted_prefix_gain=rew[i]-rew[0],
   predicted_discounted_terminal_gain=(values[i]-rew[i])-(values[0]-rew[0]),predicted_Q_member_deltas=pq.tolist(),
   ranking=s['ranking'],raw_Q_ensemble_std_mean=float(np.asarray(s['q_members']).std(0).mean()),
   discounted_value_ensemble_std_note='ranking.ensemble_std is the discounted total-return spread, not raw Q spread',
   same_batch_mean_elite_audit=s['same_batch_audit'])
  if mode in b['terminal'] and mode in d['terminal']:
   aq=np.asarray(d['terminal'][mode]['actual_state_Q_members'])-np.asarray(b['terminal'][mode]['actual_state_Q_members'])
   decomp.update(actual_state_same_action_Q_member_deltas=aq.tolist(),
    predicted_vs_actual_state_member_sign_agreement=float(np.mean(np.sign(pq)==np.sign(aq))),
    predicted_members_agree_with_actual_gain=float(np.mean(np.sign(pq)==np.sign(gain))),
    actual_state_members_agree_with_actual_gain=float(np.mean(np.sign(aq)==np.sign(gain))),
    actual_state_prior_Q_minus_SAC_RTG_baseline=float(np.mean(b['terminal']['actual_state_actual_prior_Q_members'])-b['actual_terminal_continuation']),
    actual_state_prior_Q_minus_SAC_RTG_mean=float(np.mean(d['terminal']['actual_state_actual_prior_Q_members'])-d['actual_terminal_continuation']))
  r['definitions'][mode]=decomp
 r['all_variant_search']={name:dict(metrics=b['search'][name]['metrics'],same_batch_audit=s['same_batch_audit']) for name,s in b['common_pool_scores'].items()}
 rows.append(r)
result=dict(origins=rows,unavailable=unavailable,scope='Nine diagnostic origins only, not held-out selection. Exactly H commands then SAC. Latent RMSE is dimensionless. Actual-state Q uses the same fixed terminal actions as predicted-state Q; direct Q-vs-SAC RTG difference is not purely Q error. Ranking agreement uses all ten two-Q combinations on one shared candidate pool. Search elites are rescored together with mean/baseline, not compared using changing-Q raw scores.')
(REPORT/'value_ranking_probe.json').write_text(json.dumps(result,indent=2))
for r in rows:print(r['scenario'],r['before'],'pred',round(r['definitions']['D0']['predicted_gain'],3),'actual',round(r['actual_discounted_gain'],3),'latent',r['mean_terminal_latent_rmse'])
