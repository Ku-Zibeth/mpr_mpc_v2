"""After independent full episodes, measure isolated true-cycle latency then analyze."""
import time
from comfort_study import *
while not (R/'closed_complete.json').exists():
    if 'Traceback' in (R/'pipeline.log').read_text():raise RuntimeError('Collection failed; inspect logs and keep partial data')
    time.sleep(2)
p=load(R/'protocol.json');env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
hardware=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True)
(R/'hardware_timing_start.txt').write_text(hardware.stdout+hardware.stderr)
for scene in p['scenes'][:2]:
    for group in ['A','B','C']:
        with (R/f'latency_{group}_{scene}.log').open('x') as f:
            subprocess.run([sys.executable,'-u',str(R/'closed.py'),'--group',group,'--scene',str(scene),'--seed','1','--smoke','105','--timing'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('ISOLATED_CYCLE',group,scene,flush=True)
dump(R/'latency_complete.json',dict(groups=3,maps=2,warmup=5,calls_per_case=100,actual_env_steps=True,latency_rollouts_not_extra_driving_samples=True))
for name in ['analysis.py','figures.py']:
    with (R/(name[:-3]+'.log')).open('x') as f:
        subprocess.run([sys.executable,'-u',str(R/name)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('FINISHED',name,flush=True)
