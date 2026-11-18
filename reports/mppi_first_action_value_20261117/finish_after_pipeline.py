"""Finalize only when all prerequisite measurements/gates really complete."""
import os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *
while not (REPORT/'collection_pipeline_complete.json').exists():
    log=REPORT/'pipeline.log'
    if log.exists() and 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('Pipeline failed; retain logs, fix before finalizing.')
    time.sleep(2)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (REPORT/'final_report.log').open('x') as f:
    subprocess.run([sys.executable,'-u',str(REPORT/'finalize_study.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
