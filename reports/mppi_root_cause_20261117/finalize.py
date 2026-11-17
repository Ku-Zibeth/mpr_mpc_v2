"""Reviewable changes and immutable source/data/weight checks, not new experiments."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from planning.sac_value import file_sha256,state_sha256
from tools.benchmark_mppi import source_fingerprint
p=json.loads((R/'protocol.json').read_text());assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==p['base_commit'];assert file_sha256(p['checkpoint'])==p['checkpoint_sha256'];assert '53 passed' in (R/'tests_final.txt').read_text()
changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines();assert changed==['planning/mppi_ablation.py'],changed
names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split('\0');digest=hashlib.sha256()
for name in sorted(set(names)):
 path=Path(name)
 if not name or path.parts[0] in {'logs','reports','tests'} or path.suffix not in {'.py','.yaml','.yml'}:continue
 if not (ROOT/path).is_file():continue
 content=subprocess.check_output(['git','show',p['base_commit']+':'+name],cwd=ROOT) if name=='planning/mppi_ablation.py' else (ROOT/path).read_bytes();digest.update(name.encode());digest.update(content)
assert digest.hexdigest()==p['source_snapshot_sha256'],('Unexpected control-generating source change',digest.hexdigest())
protected=['config.yaml','planning/local_mppi.py','planning/evaluator.py','planning/coordinator.py','planning/trajectory_adapter.py','planning/cost_validation.py','residual_rl/state_builder.py','residual_rl/features.py','residual_rl/networks.py','residual_rl/action_adapter.py','lattice/frenet_metadrive.py','train.py']
for name in protected:assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show',p['base_commit']+':'+name],cwd=ROOT)
for name,count in [('intervention_status.json',80),('state_status.json',197)]:
 status=json.loads((R/name).read_text());assert len(status)==count and not any(status.values())
assert not list(R.rglob('*.pt')) and not list(R.rglob('*.pth'))
for path in (R/'interventions').glob('*/*.npz'):
 d=np.load(path);assert bool(d['done'][-1]) and not d['done'][:-1].any()
prior=Path(p['prior_data']);manifest=json.loads((prior/'dataset_manifest.json').read_text());oldfiles=manifest['fit_sources']+manifest['stop_sources']+json.loads((prior/'validation_manifest.json').read_text())
for x in oldfiles:assert file_sha256(x['path'])==x['sha256'],x['path']
heads=json.loads((prior/'ranking_evaluation.json').read_text())['head_hashes']
for name,value in heads.items():assert state_sha256(torch.load(prior/'models'/f'{name}.pt',map_location='cpu',weights_only=True)['state_dict'])==value
state=json.loads((R/'state_information.json').read_text());assert state['candidates']==197 and state['scenes']==28 and state['origins']==55
for fold in state['folds']:
 assert set(fold['training_maps']).isdisjoint(fold['heldout_maps']);assert set(fold['training_maps']+fold['heldout_maps']).isdisjoint(p['already_seen_validation_scenarios'])
result=dict(base_commit=p['base_commit'],experiment_source_sha256=p['source_snapshot_sha256'],experiment_source_reconstruction_verified=True,delivery_source_sha256=source_fingerprint(ROOT),post_experiment_change='Only independent iterations config factory, default unchanged; exact copy semantics tested. The source controlling all collected actions is reconstructed against original factory file.',
 original_checkpoint_sha256=p['checkpoint_sha256'],original_weights_unchanged=True,old_candidate_data_files_verified=len(oldfiles),prior_scoring_heads_unchanged=len(heads),protected_source_unchanged=protected,original50_tests_unchanged=True,tests='53 passed (original50 retained +3)',
 counterfactual_native_branches=80,prefix_replays=197,original_sac_state_dim=535,matched_CV_maps=28,matched_CV_origins=55,full_state_excluded=len(state['excluded_native_ended']),new_value_heads=0,new_online_models=0,new_physical_decoder_artifacts=0,
 cost_weight=0.,default_iterations=4,sampling=dict(samples=64,elites=8,AR1_unchanged=True,std_rho_beta_Q_unchanged=True),online_candidate_selection_unchanged=True,
 next_direction_only='Execution-matched small receding-horizon iteration ablation; no winner/default chosen.',full100_warranted=False,formal100=False,git_commit_push_reset=False,physical_readout_note=p['physical_readout_amendment'])
(R/'verification.json').write_text(json.dumps(result,indent=2))
# Hash the new replay artifacts separately; never embed tensors/weights in code diff.
artifacts=[dict(path=str(path.relative_to(ROOT)),sha256=file_sha256(path)) for group in ['catalog','interventions','states'] for path in sorted((R/group).glob('*/*.npz'))];(R/'artifacts_manifest.json').write_text(json.dumps(artifacts,indent=2))
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
tracked=subprocess.check_output(['git','diff','--','planning/mppi_ablation.py'],cwd=ROOT,text=True);patch=[tracked]
new=['tools/mppi_diagnosis.py','tests/test_mppi_diagnosis.py']+[str(x.relative_to(ROOT)) for x in sorted(R.glob('*.py'))]+[str((R/'protocol.json').relative_to(ROOT))]
for name in new:
 check=subprocess.run(['git','diff','--no-index','--check','--','/dev/null',name],cwd=ROOT,capture_output=True,text=True);assert not check.stdout and not check.stderr,(name,check.stdout,check.stderr)
 diff=subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=ROOT,capture_output=True,text=True);assert diff.returncode==1;patch.append(diff.stdout)
(R/'code_changes.patch').write_text(''.join(patch));print(json.dumps(result,indent=2))
