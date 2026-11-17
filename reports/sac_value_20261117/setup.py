import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from tools.benchmark_mppi import source_fingerprint
from planning.sac_value import file_sha256
old=json.loads((ROOT/'reports/mppi_calibration_20261116/protocol.json').read_text());scenes=np.random.default_rng(2026111701).choice(np.arange(60000,62000),12,replace=False).tolist()
p=dict(base_commit='430e8abd886cd9759684b1de7c2352e5d3aee7a3',checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],source_environment_config=old['source_environment_config'],
 source_snapshot_sha256=source_fingerprint(ROOT),fit_scenarios=scenes[:8],early_stop_scenarios=scenes[8:],training_scenarios=scenes,
 origin_steps=[50,150],train_branches=['B','gaussian_1','gaussian_2'],window_stride=5,horizon=10,seed=1,
 development_scenarios=old['dev_scenarios'],validation_scenarios=old['validation_scenarios'],old_data=str(ROOT/'reports/mppi_calibration_20261116/probes'),
 targeted_dev_origins=[[old['dev_scenarios'][0],50],[old['dev_scenarios'][1],150],[old['dev_scenarios'][2],50],[old['dev_scenarios'][3],150]],
 acceptance=dict(min_independent_scenes=6,pair_accuracy_ci95_low_greater_than=.5,spearman_ci95_low_greater_than=0.,j_mae_less_than_old_q=True),
 training=dict(hidden=[128,64],lr=.0003,batch_size=256,max_epochs=100,patience=12,loss='normalized SmoothL1 + matched-origin pair-difference SmoothL1; fixed equal weight; epoch selection only on the four inner-stop scenes'),
 protocol='Only new60000-range scenes train the head. Real frozen SAC after H=10, native done. Gaussian prefixes are the first two pre-fixed AR(1) samples, not value-selected. Imagined terminal latent is primary input. Actual-latent head is diagnostic oracle only, never deployed. Sliding windows are real executed H-prefixes followed by frozen SAC; no invented future states.',
 output_policy='Keep baseline/mean online default. Audit all actual samples under one terminal reference using cached reward/dynamics; no extra full WM rollout or sampling. Cost weight0 unless existing validator passes. No larger driving validation if development suffix return/safety does not improve.',
 original_weights_frozen=True,head_training_authorized=True,formal100=False)
assert file_sha256(p['checkpoint'])==p['checkpoint_sha256']
assert not set(scenes)&set(p['development_scenarios']+p['validation_scenarios']+old['additional_test_scenarios'])
path=R/'protocol.json';assert not path.exists();path.write_text(json.dumps(p,indent=2));print(scenes)
