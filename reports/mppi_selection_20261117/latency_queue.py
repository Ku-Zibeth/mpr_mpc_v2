import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent;ROOT=R.parents[1]
while not (R/'stage2_analysis.json').exists() or not (R/'development_selection.json').exists():time.sleep(3)
choice=json.loads((R/'development_selection.json').read_text())
if choice['candidate']:
 while not (R/'validation_analysis.json').exists():time.sleep(3)
with (R/'latency.log').open('x') as log:subprocess.run([sys.executable,'-u',str(R/'latency.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl'))
