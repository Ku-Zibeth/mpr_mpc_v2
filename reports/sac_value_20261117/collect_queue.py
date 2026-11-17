"""Collect fixed Gaussian prefixes, keeping original WM/SAC read-only."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from tools.benchmark_mppi import source_fingerprint
from planning.sac_value import file_sha256
p=json.loads((R/'protocol.json').read_text());assert source_fingerprint(ROOT)==p['source_snapshot_sha256'];assert file_sha256(p['checkpoint'])==p['checkpoint_sha256']
assert '47 passed' in (R/'tests_final.txt').read_text()
queue=[];done={};logs=R/'collect_logs';logs.mkdir(exist_ok=True)
for s in p['training_scenarios']:
 for before in p['origin_steps']:
  for b in p['train_branches']:
   name=f'{s}_{before}_{b}';path=R/'training_data'/f'{s}_{before}'
   if (path/(b+'.npz')).exists():done[name]=0;continue
   queue.append((name,s,before,b))
running=[]
while queue or running:
 while queue and len(running)<4:
  name,s,before,b=queue.pop(0);log=(logs/(name+'.log')).open('x')
  job=subprocess.Popen([sys.executable,'-u',str(R/'collect.py'),'--scenario',str(s),'--before',str(before),'--branch',b],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
   env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl'))
  running.append((name,job,log))
 for name,job,log in running[:]:
  status=job.poll()
  if status is not None:
   running.remove((name,job,log));log.close();done[name]=status;(R/'collection_status.json').write_text(json.dumps(done,indent=2));print(len(done),'/72',name,status,flush=True)
 if running:time.sleep(2)
assert len(done)==72 and not any(done.values()),done
assert source_fingerprint(ROOT)==p['source_snapshot_sha256'];assert file_sha256(p['checkpoint'])==p['checkpoint_sha256']
