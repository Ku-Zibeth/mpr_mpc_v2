"""New predeclared maps, paired native termination, no old outcomes overwritten."""
from study import *
import time

def run(jobs):
    todo=list(jobs);active=[];done=0;(R/'logs').mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    try:
        while todo or active:
            while todo and len(active)<6:
                split,scene,seed,method=todo.pop(0);name=f'{split}_{method}_{scene}_r{seed}';f=(R/'logs'/f'{name}.log').open('x')
                child=subprocess.Popen([sys.executable,'-u',str(R/'closed.py'),'--split',split,'--scene',str(scene),'--seed',str(seed),'--method',method],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                active.append((name,child,f))
            for name,child,f in active[:]:
                code=child.poll()
                if code is None:continue
                f.close();active.remove((name,child,f));assert code==0,(name,code)
                done+=1;print('NATIVE_CLOSED_LOOP',name,done,'/',len(jobs),flush=True)
            if active:time.sleep(1)
    finally:
        for _,child,f in active:
            if child.poll() is None:child.terminate()
            child.wait();f.close()

def verify(scenes,methods,split,seeds=[1]):
    for scene in scenes:
        for seed in seeds:
            paths=[R/split/f'{m}_{scene}_r{seed}' for m in methods]
            initial=[load(x/'initial.json')['hashes'] for x in paths];policy=[load(x/'first_policy_fingerprint.json') for x in paths]
            assert all(x==initial[0] for x in initial) and all(x==policy[0] for x in policy)
            assert all(load(x/'episode.json')['native_done'] and load(x/'episode.json')['weights_unchanged'] for x in paths)
if __name__=='__main__':
    assert load(R/'contact_audit.json')['passed']
    assert '56 passed' in (R/'tests.log').read_text()
    p=load(R/'protocol.json');scenes=p['development_scenes'];methods=p['development_methods']
    dump(R/'frozen_before_development.json',dict(source_sha={str(f.relative_to(ROOT)):file_sha(f) for f in [ROOT/'planning/local_mppi.py',ROOT/'planning/coordinator.py',ROOT/'planning/native_road.py',ROOT/'planning/tracking_evaluator.py',R/'variant_configs.json',R/'protocol.json']},checkpoint_sha256=file_sha(Path(p['checkpoint']))))
    run([('development',scenes[0],1,m) for m in methods]);verify(scenes[:1],methods,'development')
    dump(R/'pilot.json',dict(scene=scenes[0],paired_initial_and_frozen_policy=True,native_termination=True))
    run([('development',s,1,m) for s in scenes[1:] for m in methods]);verify(scenes,methods,'development')
    dump(R/'development_complete.json',dict(episodes=48,independent_maps=12,paired_initial_and_frozen_policy=True))
    print('DEVELOPMENT_COMPLETE_48',flush=True)
