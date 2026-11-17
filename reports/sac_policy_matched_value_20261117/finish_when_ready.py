"""Finish offline analysis only after all33 native branches andtests complete."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent;ROOT=R.parents[1]
while True:
    f=R/'paired_status.json';status=json.loads(f.read_text()) if f.exists() else {}
    if any(status.values()):raise RuntimeError(status)
    if len(status)==33:break
    time.sleep(2)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for script,args in [('analyze_values.py',['final']),('finalize.py',[])]:
    with (R/(script+'.log')).open('x') as f:
        subprocess.run([sys.executable,'-u',str(R/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
