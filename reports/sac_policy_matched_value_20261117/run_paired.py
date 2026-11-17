"""Fresh simulator process for every native suffix; stop on any mismatch."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent;ROOT=R.parents[1];p=json.loads((R/'protocol.json').read_text())
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
logs=R/'fresh_process_logs';logs.mkdir(exist_ok=False);done={}

def run_group(queue):
    running=[]
    while queue or running:
        while queue and len(running)<3 and not any(done.values()):
            s,c,method,rep,replay=queue.pop(0);key=f'{s}_{c}_{method}_r{rep}'+('_replay' if replay else '')
            log=(logs/(key+'.log')).open('x');args=[sys.executable,'-u',str(R/'collect_paired.py'),'--scenario',str(s),'--candidate',c,'--policy',method,'--repeat',str(rep)]
            if replay:args.append('--replay')
            job=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);running.append((key,job,log))
        for key,job,log in running[:]:
            status=job.poll()
            if status is not None:
                running.remove((key,job,log));log.close();done[key]=status;(R/'paired_status.json').write_text(json.dumps(done,indent=2));print('paired suffixes',len(done),'/33',key,status,flush=True)
        if not running and any(done.values()):break
        time.sleep(1)
    assert not any(done.values()),done

run_group([(s,c,'lattice_sac',1,False) for s in p['fresh_scenarios'] for c in p['candidates']])
run_group([(s,c,method,rep,False) for s in p['fresh_scenarios'] for c in p['candidates'] for rep in p['repeat_seeds'] for method in p['takeover_policies'] if (method,rep)!=('lattice_sac',1)])
run_group([(p['fresh_scenarios'][0],'baseline','lattice_sac',1,True)])
assert len(done)==33
