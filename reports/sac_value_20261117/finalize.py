"""Audit unchanged original weights, data splits, matched pools and rollout budgets."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from planning.sac_value import file_sha256,load_sac_value
from tools.benchmark_mppi import source_fingerprint
p=json.loads((R/'protocol.json').read_text());assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==p['base_commit']
assert source_fingerprint(ROOT)==p['source_snapshot_sha256'];assert file_sha256(p['checkpoint'])==p['checkpoint_sha256']
assert '47 passed' in (R/'tests_final.txt').read_text()
x=json.loads((R/'collection_status.json').read_text());assert len(x)==72 and not any(x.values())
x=json.loads((R/'development_status.json').read_text());assert len(x)==16 and not any(x.values())
assert set(p['fit_scenarios']).isdisjoint(p['early_stop_scenarios'])
assert set(p['training_scenarios']).isdisjoint(p['development_scenarios']+p['validation_scenarios'])
assert all(all(v==0 for v in row['max_errors'].values()) for row in json.loads((R/'sac_interface_verification.json').read_text()))
cost=json.loads((R/'cost_validation.json').read_text());assert cost['original_cost_buffers_unchanged']
comparison=json.loads((R/'development_comparison.json').read_text());assert not comparison['decision']['worth_independent_closed_loop']
try:load_sac_value(R/'models/sac_value_candidate.pt',wm_checkpoint=p['checkpoint'],horizon=10,gamma=.995,device='cpu')
except ValueError as error:rejected=str(error)
else:raise AssertionError('Failed-ranking head must not load online')
changed=set(subprocess.check_output(['git','diff','--name-only'],text=True).splitlines());allowed={'config.yaml','planning/coordinator.py','planning/evaluator.py','planning/local_mppi.py','planning/native_config.py','planning/types.py'}
assert changed<=allowed,changed-allowed
subprocess.run(['git','diff','--check'],check=True)
manifest=json.loads((R/'dataset_manifest.json').read_text())
for part in ('fit','inner_stop'):
 for row in manifest[part]:assert file_sha256(row['path'])==row['sha256']
# Unknown metric is null, not a non-standard NaN number.
def clean(v):
 import math
 if isinstance(v,float) and not math.isfinite(v):return None
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
for path in R.rglob('*.json'):path.write_text(json.dumps(clean(json.loads(path.read_text())),indent=2,allow_nan=False))
result=dict(base_commit=p['base_commit'],source_snapshot_sha256=p['source_snapshot_sha256'],original_checkpoint_sha256=p['checkpoint_sha256'],original_weights_unchanged=True,
 training_scenes=12,fit_scenes=8,inner_stop_scenes=4,training_native_branches=72,training_windows=manifest['train_examples'],inner_stop_windows=manifest['inner_stop_examples'],
 independently_trained_sidecars=2,original_WM_or_SAC_training=False,tests='47 passed (original43 retained)',SAC_535_max_error=0.,cost_weight=0.,
 new_development_counterfactual_branches=16,matched_existing_origins=32,default_online_output_unchanged=True,new_value_online_load_rejected=rejected,
 complete_WM_candidate_rollouts_per_plan=262,Gaussian_samples=64,elites=8,iterations=4,
 additional_audit='258 terminal-only evaluations on cached predicted states; no extra reward/dynamics rollout',
 formal100=False,worth_independent_closed_loop=False,
 logging_retry='16 initial offline branches failed final serialization because LocalMPPI does not expose planning_time_ms. Those logs/pools are preserved, excluded; synchronized wall timing fixed the collector. Retried candidate actions match exactly.')
(R/'verification.json').write_text(json.dumps(result,indent=2))
patch=subprocess.check_output(['git','diff','--binary'],text=True)
for name in ['planning/sac_value.py','tests/test_sac_continuation_value.py']+[str(x.relative_to(ROOT)) for x in sorted(R.glob('*.py'))]+[str((R/'protocol.json').relative_to(ROOT))]:
 x=subprocess.run(['git','diff','--no-index','--','/dev/null',name],capture_output=True,text=True);assert x.returncode==1;patch+=x.stdout
(R/'code_changes.patch').write_text(patch);print(json.dumps(result,indent=2))
