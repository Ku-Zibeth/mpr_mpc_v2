"""Two bounded independent env workers; never overwrite an episode/result."""
import os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from study_io import R,ROOT,load,dump

p=load(R/'protocol.json');env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
jobs=[]
for scene in p['scenarios']:
    for seed in p['repeat_seeds']:
        for group in p['groups']:
            name=f'{group}_{scene}_r{seed}';output=R/'episodes'/name
            if (output/'episode.json').exists():continue
            if output.exists():
                archive=R/'failed_attempts'/name;archive.parent.mkdir(exist_ok=True)
                if archive.exists():archive=archive.with_name(name+'_'+str(int(time.time())))
                output.rename(archive)
            jobs.append((name,group,scene,seed))
running=[];completed=[];failed=[]
logs=R/'job_logs';logs.mkdir(exist_ok=True)
while jobs or running:
    while jobs and len(running)<p['parallel_workers'] and not failed:
        name,group,scene,seed=jobs.pop(0);log=logs/f'{name}_{int(time.time())}.log';stream=log.open('x')
        child=subprocess.Popen([sys.executable,'-u',str(R/'closed_loop.py'),'--group',group,'--scene',str(scene),'--seed',str(seed)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        running.append((name,child,stream,log))
    for name,child,stream,log in running[:]:
        status=child.poll()
        if status is None:continue
        running.remove((name,child,stream,log));stream.close()
        (completed if status==0 else failed).append(dict(name=name,status=status,log=str(log)))
        print('EPISODE',name,'status',status,'completed',len(completed),'failed',len(failed),flush=True)
    if failed and not running:break
    if running:time.sleep(1)
assert not failed,failed
dump('collection_complete.json',dict(completed=completed,total_native_episodes=32,groups=p['groups'],new_weights=0))
print('ALL32 NATIVE CLOSED LOOP EPISODES COMPLETE',flush=True)
