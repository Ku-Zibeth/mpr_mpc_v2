"""Fresh process per paired native episode; never overwrite earlier experiments."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from tools.benchmark_mppi import source_fingerprint
from planning.sac_value import file_sha256
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['stage1','stage2','validation'],required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();p=json.loads((R/'protocol.json').read_text());assert source_fingerprint(ROOT)==p['source_snapshot_sha256'];assert file_sha256(p['checkpoint'])==p['checkpoint_sha256']
phase='validation' if a.stage=='validation' else 'development';groups=['B',json.loads((R/'development_selection.json').read_text())['candidate']] if a.stage=='validation' else ['B','M1','M2','M4'] if a.stage=='stage1' else ['P','E','PE'];scenes=p[phase+'_scenarios'];queue=[];done={};running=[];attempt=0
while (R/(a.stage+('' if attempt==0 else f'_retry{attempt}')+'_logs')).exists():
 if not a.resume and attempt==0:break
 attempt+=1
logs=R/(a.stage+('' if attempt==0 else f'_retry{attempt}')+'_logs');logs.mkdir(exist_ok=True)
for scene in scenes:
 for seed in p['repeat_seeds']:
  for group in groups:
   key=f'{group}_{scene}_r{seed}';out=R/'episodes'/phase/key
   if a.resume and (out/'episode.json').exists():done[key]=0;continue
   if a.resume and out.exists():
    archive=R/'failed_collector_device'/phase/key;archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists();out.rename(archive)
   queue.append((key,group,scene,seed))
expected=len(queue)+len(done);env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl')
while queue or running:
 while queue and len(running)<4 and not any(done.values()):
  key,g,s,seed=queue.pop(0);log=(logs/(key+'.log')).open('x');job=subprocess.Popen([sys.executable,'-u',str(R/'closed_loop.py'),'--phase',phase,'--group',g,'--scenario',str(s),'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env);running.append((key,job,log))
 for key,job,log in running[:]:
  status=job.poll()
  if status is not None:
   running.remove((key,job,log));log.close();done[key]=status;temp=R/(a.stage+'_status.tmp');temp.write_text(json.dumps(done,indent=2));temp.replace(R/(a.stage+'_status.json'));print(len(done),'/',expected,key,status,flush=True)
 if any(done.values()) and not running:break
 if running:time.sleep(2)
assert len(done)==expected and not any(done.values()),done
assert source_fingerprint(ROOT)==p['source_snapshot_sha256'];assert file_sha256(p['checkpoint'])==p['checkpoint_sha256']
