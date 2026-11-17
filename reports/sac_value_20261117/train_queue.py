import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent
while True:
 path=R/'collection_status.json';x=json.loads(path.read_text()) if path.exists() else {}
 if len(x)==72:assert not any(x.values());break
 time.sleep(3)
with (R/'training.log').open('x') as f:
 p=subprocess.run([sys.executable,'-u',str(R/'fit_head.py')],stdout=f,stderr=subprocess.STDOUT,env=os.environ)
assert p.returncode==0
