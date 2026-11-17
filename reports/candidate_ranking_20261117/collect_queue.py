import json,os,subprocess,sys,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from tools.benchmark_mppi import source_fingerprint
from planning.sac_value import file_sha256
ap=argparse.ArgumentParser();ap.add_argument('--validation-only',action='store_true');ap.add_argument('--stop-only',action='store_true');args=ap.parse_args()
p=json.loads((R/'protocol.json').read_text());assert source_fingerprint(ROOT)==p['source_snapshot_sha256'];assert file_sha256(p['checkpoint'])==p['checkpoint_sha256']
queue=[];done={};running=[];logs=R/('validation_collect_logs' if args.validation_only else 'stop_collect_logs' if args.stop_only else 'collect_logs');logs.mkdir(exist_ok=True)
scenes=p['validation_scenarios'] if args.validation_only else p['new_stop_scenarios'] if args.stop_only else p['new_fit_scenarios']+p['new_stop_scenarios']+p['validation_scenarios']
for s in scenes:
 for before in p['origin_steps']:
  for b in p['branches']:
   name=f'{s}_{before}_{b}';folder=R/'data'/f'{s}_{before}'
   if (folder/(b+'.npz')).exists() or (folder/('unavailable_'+b+'.json')).exists():done[name]=0
   else:queue.append((name,s,before,b))
while queue or running:
 while queue and len(running)<4:
  name,s,before,b=queue.pop(0);log=(logs/(name+'.log')).open('x');job=subprocess.Popen([sys.executable,'-u',str(R/'collect.py'),'--scenario',str(s),'--before',str(before),'--branch',b],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl'))
  running.append((name,job,log))
 for name,job,log in running[:]:
  status=job.poll()
  if status is not None:
   running.remove((name,job,log));log.close();done[name]=status;status_path=R/('validation_collection_status.json' if args.validation_only else 'stop_collection_status.json' if args.stop_only else 'collection_status.json');temporary=status_path.with_suffix('.tmp');temporary.write_text(json.dumps(done,indent=2));temporary.replace(status_path);print(len(done),'/176',name,status,flush=True)
 if running:time.sleep(2)
assert len(done)==len(scenes)*8 and not any(done.values()),done
assert source_fingerprint(ROOT)==p['source_snapshot_sha256'];assert file_sha256(p['checkpoint'])==p['checkpoint_sha256']
