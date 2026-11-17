import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=Path(__file__).parent;p=json.loads((R/'protocol.json').read_text());env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
# All catalogs/choices are generated before any new actual suffix Return.
for s,b in p['development_origins']:
 if (R/'catalog'/f'{s}_{b}'/'catalog.npz').exists():continue
 with (R/f'prepare_{s}_{b}.log').open('x') as f:subprocess.run([sys.executable,'-u',str(R/'prepare.py'),'--scenario',str(s),'--before',str(b)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
queue=[];running=[];done={};logs=R/'intervention_logs';logs.mkdir(exist_ok=True)
for s,b in p['development_origins']:
 for protocol in ['A','B']:
  for choice in p['choices']:
   key=f'{s}_{b}_{protocol}_{choice}';queue.append((key,s,b,protocol,choice))
while queue or running:
 while queue and len(running)<4:
  key,s,b,protocol,choice=queue.pop(0);log=(logs/(key+'.log')).open('x');job=subprocess.Popen([sys.executable,'-u',str(R/'intervene.py'),'--scenario',str(s),'--before',str(b),'--protocol',protocol,'--choice',choice],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env);running.append((key,job,log))
 for key,job,log in running[:]:
  status=job.poll()
  if status is not None:
   running.remove((key,job,log));log.close();done[key]=status;temp=R/'intervention_status.tmp';temp.write_text(json.dumps(done,indent=2));temp.replace(R/'intervention_status.json');print(len(done),'/80',key,status,flush=True)
 if running:time.sleep(2)
assert len(done)==80 and not any(done.values()),done
