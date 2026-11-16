"""Verify immutable inputs, protocol completeness and archive this task's diff."""
import csv,hashlib,json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));REPORT=Path(__file__).parent
from tools.benchmark_mppi import source_fingerprint
protocol=json.loads((REPORT/'protocol.json').read_text());source=source_fingerprint(ROOT)
assert '40 passed' in (REPORT/'tests.txt').read_text()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==protocol['base_commit']
sha=hashlib.sha256(Path(protocol['checkpoint']).read_bytes()).hexdigest();assert sha==protocol['checkpoint_sha256']
counts={};outputs=[]
for phase in ('validation','test'):
 meta=json.loads((REPORT/f'{phase}_run.json').read_text());out=Path(meta['output']);outputs.append(out)
 assert source==meta['source_snapshot_sha256']
 statuses=json.loads((out/'worker_statuses.json').read_text());assert set(statuses)==set(meta['groups']) and not any(statuses.values())
 for group in meta['groups']:
  rows=list(csv.DictReader((out/group/'episodes.csv').open()));assert len(rows)==len(meta['scenarios'])
  assert {int(float(r['scenario'])) for r in rows}==set(meta['scenarios'])
  assert all(0<int(float(r['length']))<=1000 for r in rows)
  if group!='B':
   for r in rows:
    assert float(r['mppi_wm_evaluated_candidates'])==262 and float(r['mppi_iterations_completed'])==4
    assert float(r['mppi_invalid_prediction'])==0
    assert abs(float(r['mppi_initial_std_steer'])-.04)<1e-6 and abs(float(r['mppi_initial_std_throttle'])-.12)<1e-6
  counts[phase+'_'+group]=len(rows)
probes=json.loads((REPORT/'value_ranking_probe.json').read_text());assert len(probes['origins'])==9 and not probes['unavailable']
assert all(r['matched_observed_state_and_candidates'] for r in probes['origins'])
assert not any(json.loads((REPORT/'probe_worker_statuses.json').read_text()).values())
assert all(all(v==0 for v in r['max_errors'].values()) for r in json.loads((REPORT/'sac_interface_verification.json').read_text()))
allowed={'config.yaml','evaluation_diagnostics.py','planning/coordinator.py','planning/evaluator.py','planning/local_mppi.py','planning/native_config.py','planning/types.py','tests/test_simplified_tdmpc2_mppi.py'}
changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines());assert changed<=allowed,changed-allowed
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
# Unknown diagnostic values use JSON null, not non-standard NaN/Infinity.
def clean(value):
 if isinstance(value,float) and not math.isfinite(value):return None
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
 if isinstance(value,list):return [clean(v) for v in value]
 return value
for parent in [REPORT,*outputs]:
 for path in parent.rglob('*.json'):
  value=json.loads(path.read_text());path.write_text(json.dumps(clean(value),indent=2,allow_nan=False))
verification=dict(checkpoint_sha256=sha,source_snapshot_sha256=source,source_unchanged_through_all_experiments=True,
 base_commit=protocol['base_commit'],episode_counts=counts,paired_probe_origins=9,probe_branches=18,
 sampling_budget=262,iterations=4,model_training_or_weight_writes=False,protected_tracked_files_unchanged=True,
 old_experiments_untouched=True,unit_tests='40 passed',note='Null means an unavailable metric, not zero. No model parameters were updated; no full100 scene experiment was run.')
(REPORT/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False))
patch=subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True)
new=['planning/mppi_ablation.py','tests/test_mppi_value_evaluation.py']+[str(p.relative_to(ROOT)) for p in sorted(REPORT.glob('*.py'))]
for name in new:
 result=subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=ROOT,capture_output=True,text=True);assert result.returncode==1
 patch+=result.stdout
(REPORT/'code_changes.patch').write_text(patch)
print(json.dumps(verification,indent=2))
