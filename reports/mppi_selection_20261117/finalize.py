"""Frozen-source/model checks and a complete reviewable source diff."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from planning.sac_value import file_sha256,state_sha256
from tools.benchmark_mppi import source_fingerprint
p=json.loads((R/'protocol.json').read_text());assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==p['base_commit'];assert file_sha256(p['checkpoint'])==p['checkpoint_sha256'];assert source_fingerprint(ROOT)==p['source_snapshot_sha256'];assert '56 passed' in (R/'tests_final.txt').read_text();assert not list(R.rglob('*.pt')) and not list(R.rglob('*.pth'))
allowed={'planning/coordinator.py','planning/evaluator.py','planning/local_mppi.py','planning/native_config.py','planning/types.py'};changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines());assert changed==allowed,changed
protected=['config.yaml','agent.py','train.py','residual_rl/networks.py','residual_rl/state_builder.py','residual_rl/features.py','residual_rl/action_adapter.py','planning/sac_value.py','planning/candidate_value.py','planning/cost_validation.py','planning/trajectory_adapter.py','lattice/frenet_metadrive.py','vendor/tdmpc2_runtime/tdmpc2.py','vendor/tdmpc2_runtime/common/world_model.py']
for path in protected:assert (ROOT/path).read_bytes()==subprocess.check_output(['git','show',p['base_commit']+':'+path],cwd=ROOT)
for stage,count in [('stage1',32),('stage2',24)]:
 status=json.loads((R/(stage+'_status.json')).read_text());assert len(status)==count and not any(status.values())
selection=json.loads((R/'development_selection.json').read_text());assert selection['candidate'] is None and not (R/'episodes/validation').exists()
trace=json.loads((R/'trace_verification.json').read_text());assert all(trace[k] for k in ['all56_native_episodes_validated','all_control_ticks_replanned','all_selected_first_actions_match','all_nonbaseline_selections_strictly_better_predicted','complete_candidate_arrays_saved','policy_prior_padding_fixed'])
gpu=json.loads((R/'gpu_regression.json').read_text());assert gpu['SAC535_unchanged'] and gpu['coarse_SAC_WM_features_unchanged']
original=json.loads((ROOT/'reports/candidate_ranking_20261117/ranking_evaluation.json').read_text())['head_hashes']
for name,expected in original.items():assert state_sha256(torch.load(ROOT/'reports/candidate_ranking_20261117/models'/f'{name}.pt',map_location='cpu',weights_only=True)['state_dict'])==expected
metadata=[];countticks=0;currentconfigs={}
for folder in sorted((R/'episodes/development').iterdir()):
 if not (folder/'episode.json').exists():continue
 row=json.loads((folder/'episode.json').read_text());assert row['native_done'];countticks+=row['length'];metadata.append(dict(path=str((folder/'trace.npz').relative_to(ROOT)),sha256=file_sha256(folder/'trace.npz'),steps=row['length'],counts=row['counts']));currentconfigs[row['configuration']]=row['options']
assert len(metadata)==56
(R/'artifacts_manifest.json').write_text(json.dumps(metadata,indent=2))
result=dict(base_commit=p['base_commit'],source_snapshot_sha256=p['source_snapshot_sha256'],original_checkpoint_sha256=p['checkpoint_sha256'],original_models_and_config_unchanged=True,protected_files=protected,prior_scoring_heads_unchanged=len(original),new_neural_models=0,tests='56 passed (original53 retained +3)',actual_native_closed_loop_episodes=56,environment_control_ticks=countticks,development_maps=4,planner_repeat_seeds=p['repeat_seeds'],independent_validation_opened=False,independent_validation_not_run_reason='No development method qualifies; sealed6-map set untouched',all_ticks_fresh_Lattice_SAC_and_requested_MPPI=True,all_sampling_budgets_verified=True,complete_action_and_Q_trace_saved=True,
 no_extra_full_WM_rollout_for_elite=True,terminal_policy_prior='deterministic squashed TD-MPC2 actor mean, NOT original stochastic actor and NOT imaginedResidualSAC',selection='Explicit opt-in mode; legacy missing-field behavior preserved. Baseline/Mean final2 Qavg unchanged, cachedfinalElite terminal reevaluated with EXACT same Q member IDs; strict gain>0.',cost_weight=0.,default_candidate_mode_and_old_config_unchanged=True,
 gpu=torch.cuda.get_device_name(),isolated_timing_calls_per_method=100,diagnostic_cached_vs_direct_max_error=max(r['maximum_cached_vs_direct_error'] for r in gpu['rows']),SAC535_regression_max_error=0.,formal100=False,training=False,git_commit_push_reset=False,collector_retries='24 failed-before-first-env-step GPU/CPU verifier attempts archived; sampler/model unchanged. Eight completedB native episodes reused; only same24 planned episodes rerun. Initial latency import-path failure produced no timing sample; retried after correcting report-module import.')
(R/'verification.json').write_text(json.dumps(result,indent=2))
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True);patch=[subprocess.check_output(['git','diff'],cwd=ROOT,text=True)]
for name in ['tests/test_mppi_final_selection.py']+[str(x.relative_to(ROOT)) for x in sorted(R.glob('*.py'))]+[str((R/'protocol.json').relative_to(ROOT))]:
 check=subprocess.run(['git','diff','--no-index','--check','--','/dev/null',name],cwd=ROOT,capture_output=True,text=True);assert not check.stdout and not check.stderr,(name,check.stdout,check.stderr)
 diff=subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=ROOT,capture_output=True,text=True);assert diff.returncode==1;patch.append(diff.stdout)
(R/'code_changes.patch').write_text(''.join(patch));print(json.dumps(result,indent=2))
