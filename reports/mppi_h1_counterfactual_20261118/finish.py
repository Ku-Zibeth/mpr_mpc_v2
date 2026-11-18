"""After successful collection: audit, paired analysis, figures and report in that order."""
import os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
while not (R/'collection_complete.json').exists():
    log=(R/'pipeline.log').read_text()
    if 'Traceback' in log:raise RuntimeError('Collection failed: do not analyse incomplete branches as a complete experiment')
    time.sleep(2)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for script in ['audit_data.py','analyze.py','plots.py','report.py']:
    with (R/(script.removesuffix('.py')+'.log')).open('x') as stream:
        subprocess.run([sys.executable,'-u',str(R/script)],cwd=R.parents[1],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('FINISHED',script,flush=True)
