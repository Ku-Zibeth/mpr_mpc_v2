import json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from tools.mppi_diagnosis import grouped_folds
from tools.benchmark_mppi import source_fingerprint
from planning.sac_value import file_sha256
old=json.loads((ROOT/'reports/candidate_ranking_20261117/protocol.json').read_text());base='bfe035d7ea441bb845f65d7b72c3e3dbebfb3d33'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==base
fit=old['old_fit_scenarios']+old['new_fit_scenarios'];stop=old['old_stop_scenarios']+old['new_stop_scenarios']
p=dict(base_commit=base,checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],source_environment_config=old['source_environment_config'],source_snapshot_sha256=source_fingerprint(ROOT),
 prior_data=str(ROOT/'reports/candidate_ranking_20261117'),legacy_data=str(ROOT/'reports/sac_value_20261117'),
 development_origins=[[old['new_fit_scenarios'][i],50 if i%2==0 else 150] for i in range(4)],
 origin_selection='Fixed first four entries in pre-existing Fit scenario list; alternating original50/150 steps. All are already-seen development diagnostic origins, NOT fresh independent tests. No selection by actual Return. Reconstruct saved original-Q 64x4 pools exactly.',
 choices=['baseline','gaussian']+[f'{kind}{k}' for k in range(1,5) for kind in ['mean','elite']],interventions=dict(A=10,B=1),horizons=[1,3,5,10],horizon=10,gamma=.995,seed=1,
 fit_scenarios=fit,inner_stop_scenarios=stop,already_seen_validation_scenarios=old['validation_scenarios'],cv_scenarios=fit+stop,cv_folds=grouped_folds(fit+stop),
 probe=dict(model='OFFLINE pair-difference linear ridge, same effective_df16; no MLP, no persistent value head',effective_df=16,normalization='training fold only, scene-origin-candidate balanced',target='actual SAC continuation RTG atH10, not unexecuted Q(a_H); also inspect actual-prefix+predicted-continuation total ordering',tolerance=1.,full_state='original ResidualStateBuilder at actual native nonterminal H observation; never online imagined535',ablate_groups=['path','wm_return','feasibility','risk']),
 iterations_ablation=[1,2,4],bootstrap_samples=2000,bootstrap_seed=1,cost_weight=0.,online_config_unchanged=True,new_value_heads=0,formal100=False,
 q_note='Sampling remains current cfg random_pair,beta=.5. Capture exact RNG-neutral Q pairs. Uniform score reporting conditions on the same captured random pair; no comparing raw different-round Q scales. Independent1/2/4 planner calls restore initial global RNG and private sampler seed; only iterations differ.',
 physical_note='WM has no physical-state decoder. Vehicle displacement/yaw/speed errors are from existing frozen kinematic rollout, separately labelled; latent L2 is never meters. No new learned physical decoder or model adaptation.',
 distribution_note='Original Fit/InnerStop/seenValidation summaries are retrospective descriptions; only Fit+InnerStop enter grouped CV. Do not call old validation independent. No new scenario collection selected by its Return.')
assert file_sha256(p['checkpoint'])==p['checkpoint_sha256'];assert not set(p['cv_scenarios'])&set(p['already_seen_validation_scenarios']);assert len(p['cv_scenarios'])==28
path=R/'protocol.json';assert not path.exists();path.write_text(json.dumps(p,indent=2));print(p['development_origins'])
