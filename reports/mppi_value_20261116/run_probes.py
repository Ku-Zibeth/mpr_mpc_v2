"""Nine prespecified origins, two fresh-process branches each, at most two workers."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPORT=Path(__file__).parent
origins=[(1714,250),(559,600),(985,100),(2256,100),(1760,100),(539,100),(792,100),(872,100),(1719,100)]
(REPORT/'probe_protocol.json').write_text(json.dumps(dict(origins=origins,branches=['sac','mean'],actual_prefix='H commands then SAC, not H+1',additional_origins='Prespecified step100 of unchanged SAC policy; not archived D failure states',scope='Old scenes for diagnosis only. No re-tuning or selection from this analysis.'),indent=2))
queue=[(s,b,v) for s,b in origins for v in ('sac','mean')];running=[];status={}
while queue or running:
 while queue and len(running)<2:
  s,b,v=queue.pop(0);name=f'probe_{s}_{v}';log=(REPORT/(name+'.log')).open('w');output=REPORT/(name+'.json')
  if output.exists():raise FileExistsError(output)
  cmd=[sys.executable,'-u',str(REPORT/'probe_value_ranking.py'),'--scenario',str(s),'--before',str(b),'--variant',v,'--output',str(output)]
  proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl'),stdout=log,stderr=subprocess.STDOUT)
  running.append((name,proc,log))
 for name,proc,log in running[:]:
  result=proc.poll()
  if result is not None:
   log.close();running.remove((name,proc,log));status[name]=result
   (REPORT/'probe_worker_statuses.json').write_text(json.dumps(status,indent=2));print(name,result,flush=True)
 if running:time.sleep(2)
assert not any(status.values()),status
