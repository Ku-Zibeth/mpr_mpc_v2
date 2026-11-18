"""Wait for all native episodes, then run isolated probes sequentially."""
import os,sys,time,subprocess
from study_io import R,ROOT,load,dump
while not (R/'D_collection_complete.json').exists():time.sleep(2)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
for scene in load(R/'protocol.json')['scenarios'][:2]:
    for group in ['B','C','D']:
        with (R/f'probe_{group}_{scene}.log').open('x') as f:
            subprocess.run([sys.executable,'-u',str(R/'latency_probe.py'),'--group',group,'--scene',str(scene)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('PROBE',group,scene,flush=True)
dump('probe_complete.json',dict(probes=6,calls=300,warmup_calls=30,isolated=True))
