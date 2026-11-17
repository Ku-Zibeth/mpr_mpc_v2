import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=Path(__file__).parent;env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')

def complete(stage,n):
 try:
  d=json.loads((R/(stage+'_status.json')).read_text());return len(d)==n and not any(d.values())
 except (FileNotFoundError,json.JSONDecodeError):return False

def run(script,*args,log):
 with (R/log).open('x') as f:subprocess.run([sys.executable,'-u',str(R/(script+'.py')),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
 print(script,args,'complete',flush=True)
while not complete('stage1',32):time.sleep(3)
run('analyze','--stage','stage1',log='stage1_analysis.log')
run('run_stage','--stage','stage2',log='stage2.log')
run('analyze','--stage','stage2',log='stage2_analysis.log')
selection=json.loads((R/'development_selection.json').read_text())
if selection['candidate']:
 run('run_stage','--stage','validation',log='validation.log');run('analyze','--stage','validation',log='validation_analysis.log')
else:print('Validation remains sealed: no qualified development candidate.',flush=True)
