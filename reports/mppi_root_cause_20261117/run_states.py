import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=Path(__file__).parent;p=json.loads((R/'protocol.json').read_text());prior=Path(p['prior_data']);old=Path(p['legacy_data']);manifest=json.loads((prior/'dataset_manifest.json').read_text());paths=sorted({x['path'] for key in ['fit_sources','stop_sources'] for x in manifest[key]});assert len(paths)==197
queue=[];running=[];done={}
for path in paths:
 source=Path(path);key=source.parent.name+'_'+source.stem
 if (R/'states'/source.parent.name/(source.stem+'.npz')).exists() and (R/'states'/source.parent.name/(source.stem+'.json')).exists():done[key]=0
 else:queue.append(path)
logs=R/'state_logs';logs.mkdir(exist_ok=True);env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
while queue or running:
 while queue and len(running)<4:
  source=queue.pop(0);s=Path(source);key=s.parent.name+'_'+s.stem;log=(logs/(key+'.log')).open('x');job=subprocess.Popen([sys.executable,'-u',str(R/'extract_state.py'),'--source',source],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env);running.append((key,job,log))
 for key,job,log in running[:]:
  status=job.poll()
  if status is not None:
   running.remove((key,job,log));log.close();done[key]=status;temp=R/'state_status.tmp';temp.write_text(json.dumps(done,indent=2));temp.replace(R/'state_status.json');print(len(done),'/197',key,status,flush=True)
 if running:time.sleep(2)
assert len(done)==197 and not any(done.values()),done
