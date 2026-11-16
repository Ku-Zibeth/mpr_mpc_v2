"""Two prespecified, same-origin probes; no runtime ranking gate or controller."""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
rows=[]
for scenario in (1714,559):
 b=json.loads((ROOT/f'probe_{scenario}_sac.json').read_text());d=json.loads((ROOT/f'probe_{scenario}_mean.json').read_text())
 assert b['observed_state_id']==d['observed_state_id']
 for key in ('observation','baseline','mean','predicted_values'):
  assert np.array_equal(b[key],d[key]),(scenario,key)
 assert b['decomposition']==d['decomposition']
 assert b['episode_complete'] and d['episode_complete']
 H=len(b['baseline'])-1;gamma=b['gamma']
 for value in (b,d):
  assert abs(value['actual_discounted_return']-value['actual_first_H_discounted_reward']-gamma**H*value['actual_terminal_continuation'])<1e-7
 decomposition=b['decomposition'];members=np.asarray(decomposition['terminal_Q_members']);target=np.asarray(decomposition['terminal_q'])
 matches=[pair for pair in combinations(range(len(members)),2) if np.max(np.abs(members[list(pair)].mean(0)-target))<5e-5]
 pair=matches[0] if len(matches)==1 else None
 def true_q(value):
  if value['true_state_terminal_Q_members'] is None:return None
  return float(np.asarray(value['true_state_terminal_Q_members'])[list(pair)].mean()) if pair else float(np.mean(value['true_state_terminal_Q_members']))
 rows.append(dict(scenario=scenario,before_step=b['before'],observed_state_and_candidates_identical=True,
  predicted_gain=b['predicted_gain'],actual_discounted_gain=d['actual_discounted_return']-b['actual_discounted_return'],
  predicted_prefix_reward_gain=decomposition['discounted_reward'][1]-decomposition['discounted_reward'][0],
  actual_prefix_reward_gain=d['actual_first_H_discounted_reward']-b['actual_first_H_discounted_reward'],
  predicted_discounted_terminal_gain=decomposition['discounted_terminal_contribution'][1]-decomposition['discounted_terminal_contribution'][0],
  actual_discounted_terminal_gain=gamma**H*(d['actual_terminal_continuation']-b['actual_terminal_continuation']),
  terminal_Q_member_deltas=decomposition['terminal_Q_member_deltas'],inferred_Q_pair=pair,
  baseline_true_state_Q=true_q(b),mean_true_state_Q=true_q(d),
  baseline_true_terminal_return=b['actual_terminal_continuation'],mean_true_terminal_return=d['actual_terminal_continuation'],
  baseline_terminal_latent_rmse=b['latent_rollout_rmse'][-1],mean_terminal_latent_rmse=d['latent_rollout_rmse'][-1],
  baseline_success=b['success'],mean_success=d['success'],baseline_crash=b['crash'],mean_crash=d['crash']))
result=dict(origins=rows,scope='Two prespecified origins only, not global ranking validation. Same observed ego/collected traffic/observation/candidates; not a proof of all hidden simulator states identical. H+1 actual candidate commands then a common SAC continuation. Latent RMSE is dimensionless, not physical displacement. True-state Q vs SAC return also contains continuation-policy mismatch. No runtime approval gate is added.')
(ROOT/'value_ranking_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
