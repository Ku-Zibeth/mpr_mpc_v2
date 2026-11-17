import os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent;ROOT=R.parents[1]
while not (R/'relative_dynamics.json').exists():time.sleep(3)
with (R/'physical_readout.log').open('x') as log:subprocess.run([sys.executable,'-u',str(R/'physical_readout.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
