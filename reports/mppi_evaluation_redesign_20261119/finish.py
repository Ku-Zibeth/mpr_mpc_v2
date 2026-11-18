"""Collect isolated timings after108 complete episodes; then aggregate. No automatic Git operations."""
import subprocess,time
from study import *
while not (R/'closed_complete.json').exists():
    if 'Traceback' in (R/'pipeline.log').read_text():raise RuntimeError('Collection failed, keep incomplete data and inspect technical error')
    time.sleep(2)
p=load(R/'protocol.json');env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
for scene in p['scenes'][:2]:
    for group in ['A','B','C']:
        with (R/f'probe_{group}_{scene}.log').open('x') as stream:
            subprocess.run([sys.executable,'-u',str(R/'latency_probe.py'),'--group',group,'--scene',str(scene)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('ISOLATED',group,scene,flush=True)
for scene in p['scenes'][:2]:
    states=[load(R/f'probe_{g}_{scene}.json')['initial_hashes'] for g in ['A','B','C']]
    assert states[0]==states[1]==states[2]
dump(R/'probe_complete.json',dict(calls=300,warmup=30,single_worker=True,same_state=True))
for name in ['summarize.py','figures.py']:
    with (R/(name.removesuffix('.py')+'.log')).open('x') as stream:
        subprocess.run([sys.executable,'-u',str(R/name)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('FINISHED',name,flush=True)
