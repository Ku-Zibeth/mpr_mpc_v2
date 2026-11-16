"""Bounded experiment queue. Historical results and checkpoints are never written."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from tools.benchmark_mppi import source_fingerprint
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['probes','dev','validation','test'],required=True);p.add_argument('--jobs',type=int,default=2);p.add_argument('--resume',action='store_true');a=p.parse_args();assert 1<=a.jobs<=4
meta=json.loads((R/'protocol.json').read_text());fingerprint=source_fingerprint(ROOT);assert fingerprint==meta['source_snapshot_sha256']
assert '43 passed' in (R/'tests_final.txt').read_text();assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['checkpoint_sha256']
queue=[]
if a.kind=='probes':
 for split in ('dev','validation'):
  for s in meta[split+'_scenarios']:
   for before in meta['origin_steps']:
    for b in ['B',*meta['groups']]+(['elite_Q1'] if s in meta['elite_scenarios'] else []):
     path=R/'probes'/f'{split}_{s}_{before}'/(b+'.json')
     if path.exists() or (path.parent/('unavailable_'+b+'.json')).exists() or (path.parent/'unavailable.json').exists():continue
     queue.append((f'{split}_{s}_{before}_{b}',[str(R/'probe.py'),'--scenario',str(s),'--before',str(before),'--branch',b]))
 for s,before in ((1714,250),(559,600)):
  for b in ('B','legacy_mean'):
   queue.append((f'diagnosis_{s}_{b}',[str(R/'probe.py'),'--scenario',str(s),'--before',str(before),'--branch',b,'--legacy']))
else:
 groups=['B',*meta['groups']]
 if a.kind=='test':
  selection=json.loads((R/'selection.json').read_text());assert selection.get('validation_gate_passed');groups=['B','Q1',selection['candidate']]
 scenes=meta['additional_test_scenarios'] if a.kind=='test' else meta['closed_loop_'+a.kind+'_scenarios']
 for s in scenes:
  for g in groups:queue.append((f'{a.kind}_{g}_{s}',[str(R/'closed_loop.py'),'--phase',a.kind,'--group',g,'--scenario',str(s)]))
statusfile=R/(a.kind+'_worker_statuses.json');assert a.resume or not statusfile.exists();running=[];statuses=json.loads(statusfile.read_text()) if a.resume else {};total=len(queue)+len(statuses)
logs=R/'worker_logs';logs.mkdir(exist_ok=True);print(a.kind,total,'jobs',flush=True)
while queue or running:
 while queue and len(running)<a.jobs:
  name,args=queue.pop(0);log=(logs/(name+'.log')).open('x');proc=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
   env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1'))
  running.append((name,proc,log))
 for name,proc,log in running[:]:
  ret=proc.poll()
  if ret is not None:
   log.close();running.remove((name,proc,log));statuses[name]=ret;statusfile.write_text(json.dumps(statuses,indent=2));print(len(statuses),'/',total,name,ret,flush=True)
 if running:time.sleep(2)
assert not any(statuses.values()),statuses
assert source_fingerprint(ROOT)==fingerprint
assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['checkpoint_sha256']
