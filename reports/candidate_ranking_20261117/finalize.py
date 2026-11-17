"""Read-only source/weight/data integrity checks and reviewable new-code diff."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from planning.sac_value import file_sha256,state_sha256
from tools.benchmark_mppi import source_fingerprint
p=json.loads((R/'protocol.json').read_text());assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==p['base_commit'];assert file_sha256(p['checkpoint'])==p['checkpoint_sha256'];assert source_fingerprint(ROOT)==p['source_snapshot_sha256']
assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip(),'Existing tracked files unexpectedly modified.'
assert '50 passed' in (R/'tests.txt').read_text()
manifest=json.loads((R/'dataset_manifest.json').read_text());val=json.loads((R/'ranking_evaluation.json').read_text());coverage=json.loads((R/'data_coverage.json').read_text());search=json.loads((R/'search_diagnostics.json').read_text())
for x in manifest['fit_sources']+manifest['stop_sources']:
 assert file_sha256(x['path'])==x['sha256']
for f in R.glob('data/*/*.npz'):
 data=np.load(f)
 if f.name!='audit.npz':assert bool(data['done'][-1]) and not data['done'][:-1].any()
original=json.loads((Path(p['old_data'])/'protocol.json').read_text());oldhead=torch.load(Path(p['old_data'])/'models/sac_value_candidate.pt',map_location='cpu',weights_only=True);assert state_sha256(oldhead['state_dict'])=='a033460dc9d9445b173b741192204b61d6b52298c8a9ca67fa263f727aa8f433'
for group in ['new_fit_scenarios','new_stop_scenarios','validation_scenarios']:
 others=[s for g in ['old_fit_scenarios','old_stop_scenarios','old_development_scenarios','previously_used_validation_scenarios','new_fit_scenarios','new_stop_scenarios','validation_scenarios'] if g!=group for s in p[g]]
 assert not set(p[group])&set(others)
# Preserve the exact pre-existing47 test files/implementation; only additive files.
for name in ['config.yaml','planning/local_mppi.py','planning/evaluator.py','planning/coordinator.py','planning/sac_value.py','residual_rl/networks.py','residual_rl/action_adapter.py','tests/test_residual_rl.py','tests/test_sac_continuation_value.py']:
 path=ROOT/name;assert path.read_bytes()==subprocess.check_output(['git','show',p['base_commit']+':'+name],cwd=ROOT)
validation_items=[]
for scene in p['validation_scenarios']:
 for before in p['origin_steps']:
  folder=R/'data'/f'{scene}_{before}'
  for b in p['branches']:
   path=folder/(b+'.npz')
   if path.exists():validation_items.append(dict(path=str(path),sha256=file_sha256(path)))
(R/'validation_manifest.json').write_text(json.dumps(validation_items,indent=2))
result=dict(base_commit=p['base_commit'],source_snapshot_sha256=p['source_snapshot_sha256'],original_checkpoint_sha256=p['checkpoint_sha256'],original_checkpoint_unchanged=True,existing_tracked_source_and_tests_unchanged=True,old_value_state_sha256=state_sha256(oldhead['state_dict']),old_window_reconstruction_max_error=json.loads((R/'data_audit.json').read_text())['reconstruction_max_error'],
 original_training_native_branches=69,original_planned_tasks=72,original_unreachable_tasks=3,fit_scenes=coverage['fit']['scenes'],inner_stop_scenes=coverage['inner_stop']['scenes'],fresh_validation_scenes=coverage['validation']['scenes'],fit_origins=coverage['fit']['origins'],fit_candidate_branches=coverage['fit']['raw_branches'],new_value_heads=5,online_enabled=False,independent_ranking_passed=val['validation']['passed'],formal100=False,original_WM_Q_SAC_training=False,cost_weight=0.,sampling_budget=dict(samples=64,elites=8,iterations=4),tests='50 passed (original47 retained +3)',new_CEM_or_safety_mode=False,
 new_data_native_branches=len(list((R/'data').glob('*/*_meta.json'))),new_unreachable_tasks=len(list((R/'data').glob('*/unavailable_*.json'))),new_data_generation='four reward-blind old-Q candidate definitions; no actual-return selection',validation_analysis_passes=1,all_cached_updates_valid=search['all_updates_valid'])
(R/'verification.json').write_text(json.dumps(result,indent=2))
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
# Include new executable sources and fixed protocol; binary/data artifacts remain
# separately available with manifests rather than embedding them in the patch.
names=['planning/candidate_value.py','tests/test_candidate_value.py']+[str(f.relative_to(ROOT)) for f in sorted(R.glob('*.py'))]+[str((R/'protocol.json').relative_to(ROOT))]
patch=[]
for name in names:
 check=subprocess.run(['git','diff','--no-index','--check','--','/dev/null',name],cwd=ROOT,capture_output=True,text=True);assert not check.stdout and not check.stderr,(name,check.stdout,check.stderr)
 diff=subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=ROOT,capture_output=True,text=True);assert diff.returncode==1;patch.append(diff.stdout)
(R/'code_changes.patch').write_text(''.join(patch));print(json.dumps(result,indent=2))
