"""Dev smoke before untouched-map A/B/C; no weight tuning, native full episodes."""
import time,subprocess
from comfort_study import *

def run(jobs,label):
    active=[];todo=list(jobs);finished=0;(R/'logs').mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    try:
        while todo or active:
            while todo and len(active)<6:
                name,args=todo.pop(0);stream=(R/'logs'/f'{name}.log').open('x')
                child=subprocess.Popen([sys.executable,'-u',str(R/'closed.py'),*map(str,args)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT);active.append((name,child,stream))
            for name,child,stream in active[:]:
                status=child.poll()
                if status is None:continue
                stream.close();active.remove((name,child,stream));assert status==0,(name,status)
                finished+=1;print(label,name,finished,'/',len(jobs),flush=True)
            if active:time.sleep(1)
    finally:
        for _,child,stream in active:
            if child.poll() is None:child.terminate()
            child.wait();stream.close()

def main():
    (R/'logs').mkdir(exist_ok=True)
    p=load(R/'protocol.json');jobs=[]
    for scene in p['development_smoke']['scenes']:
        for g in ['A','B','C']:jobs.append((f'dev_{g}_{scene}', ['--group',g,'--scene',scene,'--seed',1,'--smoke',100]))
    run(jobs,'DEVELOPMENT_SMOKE')
    for scene in p['development_smoke']['scenes']:
        paths=[R/'development'/f'{g}_{scene}_r1' for g in ['A','B','C']]
        states=[load(x/'initial.json')['hashes'] for x in paths];assert states[0]==states[1]==states[2]
        aa=np.load(paths[0]/'trace.npz');old=np.load(OLD/'baselines'/f'{scene}_r1'/'trace.npz')
        assert np.array_equal(aa['actions'],old['actions'][:100]) and np.array_equal(aa['observations'],old['observations'][:101])
        assert np.array_equal(aa['rewards'],old['rewards'][:100])
    dump(R/'development_smoke_passed.json',dict(episodes=6,steps=600,baseline_original_actions_observations_rewards_bitwise=True,initial_conditions_exact=True,no_weight_retuning=True))
    # Core is frozen after dev tests, BEFORE any independent test execution.
    files=['planning/local_mppi.py','planning/coordinator.py','planning/comfort_evaluator.py']
    dump(R/'core_frozen.json',dict(files={f:file_sha(ROOT/f) for f in files},comfort_config_sha256=file_sha(R/'comfort_config.json'),protocol_sha256=file_sha(R/'protocol.json')))
    jobs=[]
    for scene in p['scenes']:
        for seed in p['seeds']:
            for g in ['A','B','C']:jobs.append((f'closed_{g}_{scene}_r{seed}',['--group',g,'--scene',scene,'--seed',seed]))
    run(jobs,'INDEPENDENT_CLOSED')
    dump(R/'closed_complete.json',dict(episodes=108,maps=12,seeds=[1,2,3],native_done=True))
    print('COMFORT_ALL108_COMPLETE',flush=True)
if __name__=='__main__':main()
