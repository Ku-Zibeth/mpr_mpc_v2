import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=Path(__file__).parent

def ready(name,count):
 try:
  d=json.loads((R/name).read_text());return len(d)==count and not any(d.values())
 except (json.JSONDecodeError,FileNotFoundError):return False

def run(script):
 with (R/(script+'.log')).open('x') as log:subprocess.run([sys.executable,'-u',str(R/(script+'.py'))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
 print(script,'complete',flush=True)
while not ready('state_status.json',197):time.sleep(3)
run('probe_states');run('analyze_dynamics');run('physical_readout')
while not ready('intervention_status.json',80):time.sleep(3)
run('analyze_iterations');run('plot_results')
