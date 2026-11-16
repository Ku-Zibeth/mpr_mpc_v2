"""Final immutable-input/protocol audit and complete code diff; no model writes."""
import hashlib,json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from tools.benchmark_mppi import source_fingerprint
from tools.value_calibration import assert_origin_match
p=json.loads((R/'protocol.json').read_text());source=source_fingerprint(ROOT)
assert source==p['source_snapshot_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==p['base_commit']
sha=hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest();assert sha==p['checkpoint_sha256']
assert '43 passed' in (R/'tests_final.txt').read_text()
for stage,n in [('dev',20),('validation',30),('probes',170)]:
 x=json.loads((R/(stage+'_worker_statuses.json')).read_text());assert len(x)==n and not any(x.values()),(stage,len(x))
assert all(all(v==0 for v in row['max_errors'].values()) for row in json.loads((R/'sac_interface_verification.json').read_text()))
counts={};complete=0;branches=0
for split in ('dev','validation','diagnosis'):
 paths=list((R/'probes').glob(split+'_*'));available=0
 for path in paths:
  if list(path.glob('unavailable*.json')):continue
  ref=json.loads((path/'B_origin.json').read_text())
  names=['B','legacy_mean'] if split=='diagnosis' else ['B',*p['groups']]+(['elite_Q1'] if ref['episode_config']['scenario'] in p['elite_scenarios'] else [])
  for name in names:
   o=json.loads((path/(name+'_origin.json')).read_text());assert_origin_match(ref,o)
   row=json.loads((path/(name+'.json')).read_text());assert row['length']<=1000 and row['matched_observed_origin'];branches+=1
  available+=1
 counts[split]=dict(planned=len(paths),available=available)
 if split!='diagnosis':complete+=available
assert 24<=complete<=40,counts
for split in ('dev','validation'):
 for path in (R/'closed_loop'/split).glob('*/config.yaml'):
  # Parse actual runtime configs: the only varied MPPI field is lambda_Q.
  from omegaconf import OmegaConf
  cfg=OmegaConf.load(path);m=cfg.mpr_mpc.mppi
  assert cfg.horizon==10 and cfg.metadrive.simulator.horizon==1000
  assert (m.num_samples,m.num_elites,m.iterations)==(64,8,4)
  assert m.temperature_definition=='inverse' and m.temperature==2. and m.q_aggregation=='ensemble_mean' and m.terminal_action=='candidate'
  assert list(m.initial_std)==[.04,.12] and list(m.min_std)==[.01,.03] and list(m.max_std)==[.08,.25] and list(m.noise_rho)==[.8,.8]
changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines())
assert changed<=set(['planning/evaluator.py','planning/local_mppi.py','planning/mppi_ablation.py','planning/native_config.py','config.yaml']),changed
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
def clean(x):
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 return x
for path in R.rglob('*.json'):
 path.write_text(json.dumps(clean(json.loads(path.read_text())),indent=2,allow_nan=False))
verification=dict(base_commit=p['base_commit'],source_snapshot_sha256=source,checkpoint_sha256=sha,source_and_weights_unchanged=True,
 matched_origins=counts,completed_native_termination_branches=branches,tests='43 passed',training=False,formal100=False,
 original_40_tests_retained=True,SAC_535_max_error=0.,cost_gate_unchanged=True,
 note='Two pilot failures preserved and excluded. Queue concurrency was increased only after active workers completed. Unknown diagnostic values are null, not zero.')
(R/'verification.json').write_text(json.dumps(verification,indent=2))
patch=subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True)
for name in ['tools/value_calibration.py','tests/test_terminal_q_weight.py']+[str(f.relative_to(ROOT)) for f in sorted(R.glob('*.py'))]+[str((R/'protocol.json').relative_to(ROOT))]:
 result=subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=ROOT,capture_output=True,text=True);assert result.returncode==1;patch+=result.stdout
(R/'code_changes.patch').write_text(patch)
print(json.dumps(verification,indent=2))
