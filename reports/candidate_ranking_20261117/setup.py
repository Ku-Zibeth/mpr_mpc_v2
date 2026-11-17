import json,sys,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from planning.sac_value import file_sha256
from tools.benchmark_mppi import source_fingerprint
old=json.loads((ROOT/'reports/sac_value_20261117/protocol.json').read_text())
scenes=np.random.default_rng(2026111702).choice(np.arange(62000,65000),22,replace=False).tolist()
p=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],source_environment_config=old['source_environment_config'],source_snapshot_sha256=source_fingerprint(ROOT),
 old_data=str(ROOT/'reports/sac_value_20261117'),old_fit_scenarios=old['fit_scenarios'],old_stop_scenarios=old['early_stop_scenarios'],new_fit_scenarios=scenes[:12],new_stop_scenarios=scenes[12:16],validation_scenarios=scenes[16:],
 old_development_scenarios=old['development_scenarios'],previously_used_validation_scenarios=old['validation_scenarios'],origin_steps=[50,150],branches=['baseline','gaussian','mean','best_elite'],horizon=10,gamma=.995,seed=1,
 pair_tolerance=1.,difference_weight=.25,difference_bands=[0,1,5,20,None],action_rmse_bands=[0,.02,.05,.1,None],bootstrap_samples=2000,bootstrap_seed=1,
 models=dict(absolute_regression=dict(input_kind='terminal',hidden=[128,64],loss='absolute_SmoothL1'),absolute_plus_difference=dict(input_kind='terminal',hidden=[128,64],loss='absolute_SmoothL1 + 1.0 balanced total-return difference SmoothL1'),rank_terminal=dict(input_kind='terminal',hidden=[128,64],loss='logistic_rank + .25 normalized total_difference SmoothL1'),rank_context=dict(input_kind='context_terminal',hidden=[68,64],loss='logistic_rank + .25 normalized total_difference SmoothL1'),rank_context_actions=dict(input_kind='context_terminal_actions',hidden=[66,64],loss='logistic_rank + .25 normalized total_difference SmoothL1')),
 training=dict(lr=.0003,epochs=150,patience=20,batch_origins='all_predeclared_origins',seed=1,early_stop='same model objective on predesignated inner-stop maps only',primary='rank_context_actions',normalization='fit scenes only; balanced scene then origin then candidate',sample_unit='one actual branch at the matched origin; no new sliding-window density',near_ties='abs real total discounted-return difference <=1.0; excluded only from binary rank loss, retained for difference regression; fixed before new labels'),
 acceptance=dict(min_independent_scenes=6,pair_accuracy_ci95_low_gt=.5,spearman_ci95_low_gt=0,paired_regret_improvement_ci95_low_gt=0,baseline_mean_accuracy_not_worse=True,negative_return_selection_rate_not_worse=True),
 cost_weight=0.,formal100=False,online_scorer_enabled=False,
 collection='Candidates generated BEFORE observing any suffix rewards: original Q Gaussian/AR(1) MPPI baseline, first fixed Gaussian in iteration0, weighted final mean, actual global elite after common cached Q rescore. 64/8/4 and all cfg noise/std/rho unchanged. Native H=10 prefix then unchanged frozen SAC to native done. Same-origin RNG/config/observed state/candidates must match across fresh-process branches. No selection by actual return; unreachable origins explicitly retained.',
 isolation='New 12fit+4stop+6fresh validation maps predeclared. Old 8fit/4stop reused in their original roles only. Previous6 validation and old20/official tests never used in training, selection or fresh validation. Single validation pass after all five models and epochs freeze. No larger closed-loop unless independently qualified.')
assert p['base_commit']=='4f17fce15cca43edac7edeb253958cff979d85ba'
assert file_sha256(p['checkpoint'])==p['checkpoint_sha256']
path=R/'protocol.json';assert not path.exists();path.write_text(json.dumps(p,indent=2));print(scenes)
