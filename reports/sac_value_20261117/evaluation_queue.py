import os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent
while not (R/'dataset_manifest.json').exists():time.sleep(3)
with (R/'existing_evaluation.log').open('x') as f:
 p=subprocess.run([sys.executable,'-u',str(R/'evaluate_existing.py')],stdout=f,stderr=subprocess.STDOUT,env=os.environ)
assert p.returncode==0
