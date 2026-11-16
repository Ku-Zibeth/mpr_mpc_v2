"""Collect the predeclared validation groups without inspecting their outcomes."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent
while True:
 p=R/'dev_worker_statuses.json'
 statuses=json.loads(p.read_text()) if p.exists() else {}
 if len(statuses)==20:
  assert not any(statuses.values());break
 time.sleep(5)
with (R/'validation_run.log').open('x') as f:
 result=subprocess.run([sys.executable,'-u',str(R/'run.py'),'--kind','validation','--jobs','2'],stdout=f,stderr=subprocess.STDOUT,env=os.environ)
assert result.returncode==0
