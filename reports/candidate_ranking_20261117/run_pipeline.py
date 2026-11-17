"""Wait for prespecified data, fit once, unseal validation once, then report."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=Path(__file__).parent;p=json.loads((R/'protocol.json').read_text())

def ready(scenes):
 for s in scenes:
  for before in p['origin_steps']:
   folder=R/'data'/f'{s}_{before}'
   for b in p['branches']:
    if not ((folder/(b+'.npz')).exists() and (folder/(b+'_meta.json')).exists()) and not (folder/('unavailable_'+b+'.json')).exists():return False
 return True

def run(script):
 products={'fit':'dataset_manifest.json','evaluate':'ranking_evaluation.json','coverage':'data_coverage.json','counterfactual':'counterfactual_results.json','inspect_search':'search_diagnostics.json'}
 if (R/products[script]).exists():
  print(script,'already complete; retained',flush=True);return
 path=R/(script+'.log');attempt=0
 while path.exists():
  attempt+=1;path=R/(script+f'.retry{attempt}.log')
 with path.open('x') as f:subprocess.run([sys.executable,'-u',str(R/(script+'.py'))],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl'))
 print(script,'complete',flush=True)
while not ready(p['new_fit_scenarios']+p['new_stop_scenarios']):time.sleep(5)
run('fit')
while not ready(p['validation_scenarios']):time.sleep(5)
run('evaluate');run('coverage');run('counterfactual')
# Inspect timing after the collectors cease GPU work, not during shared load.
while True:
 try:
  if len(json.loads((R/'collection_status.json').read_text()))>=176:break
 except (json.JSONDecodeError,FileNotFoundError):pass
 time.sleep(3)
run('inspect_search')
print('Analysis complete; inspect acceptance evidence before any closed-loop expansion.',flush=True)
