import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent
while not (R/'dataset_manifest.json').exists():time.sleep(3)
p=json.loads((R/'protocol.json').read_text());queue=[(s,b,n) for s,b in p['targeted_dev_origins'] for n in ('baseline','mean','best_elite','final_elite')];running=[];statuses={};logs=R/'development_logs';logs.mkdir(exist_ok=True)
while queue or running:
 while queue and len(running)<4:
  s,b,n=queue.pop(0);name=f'{s}_{b}_{n}';log=(logs/(name+'.log')).open('x')
  job=subprocess.Popen([sys.executable,'-u',str(R/'development_probe.py'),'--scenario',str(s),'--before',str(b),'--choice',n],stdout=log,stderr=subprocess.STDOUT,
   env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl'))
  running.append((name,job,log))
 for name,job,log in running[:]:
  ret=job.poll()
  if ret is not None:
   running.remove((name,job,log));log.close();statuses[name]=ret;(R/'development_status.json').write_text(json.dumps(statuses,indent=2));print(name,ret,flush=True)
 if running:time.sleep(2)
assert not any(statuses.values()),statuses
