"""Run the ONE preregistered D correction after all32 B/C native episodes."""
import os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from study_io import R,ROOT,load,dump
while not (R/'collection_complete.json').exists():
    if 'Traceback' in (R/'batch.log').read_text():raise RuntimeError('B/C queue failed; do not continue D')
    time.sleep(2)
p=load(R/'D_protocol.json');jobs=[(s,seed) for s in p['scenarios'] for seed in p['repeat_seeds']]
while not (R/'D_preflight.json').exists():time.sleep(1)
assert load(R/'D_preflight.json')['passed'], 'D must pass unit/parity checks first'
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
running=[];done=[]
while jobs or running:
    while jobs and len(running)<2:
        scene,seed=jobs.pop(0);name=f'D_{scene}_r{seed}'
        if (R/'episodes'/name/'episode.json').exists():done.append(name);continue
        stream=(R/'job_logs'/f'{name}.log').open('x')
        process=subprocess.Popen([sys.executable,'-u',str(R/'closed_loop.py'),'--group','D','--scene',str(scene),'--seed',str(seed)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        running.append((name,process,stream))
    for name,process,stream in running[:]:
        status=process.poll()
        if status is None:continue
        running.remove((name,process,stream));stream.close();assert status==0,name
        done.append(name);print('D EPISODE',name,len(done),'/16',flush=True)
    if running:time.sleep(1)
dump('D_collection_complete.json',dict(episodes=done,expected=16,only_one_factor=True))
print('ALL16 D native episodes complete',flush=True)
