"""Frozen36-case screen, strict pilot, same-state H1 and complete native closed-loop phases."""
import subprocess,time
from study import *

def run(jobs,label):
    active=[];queue=list(jobs);done=[];(R/'logs').mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    try:
        while queue or active:
            while queue and len(active)<load(R/'protocol.json')['worker_limit']:
                name,argv=queue.pop(0)
                stream=(R/'logs'/f'{name}.log').open('x');child=subprocess.Popen([sys.executable,'-u',*map(str,argv)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,child,stream))
            for name,child,stream in active[:]:
                status=child.poll()
                if status is None:continue
                active.remove((name,child,stream));stream.close();assert status==0,(label,name,status)
                done.append(name);print(label,name,len(done),'/',len(jobs),flush=True)
            if active:time.sleep(1)
    finally:
        for _,child,stream in active:
            if child.poll() is None:child.terminate()
            child.wait();stream.close()

def closed(s,seed,g):return f'closed_{g}_{s}_r{seed}',[R/'closed.py','--group',g,'--scene',s,'--seed',seed]
def pool(s,seed):return f'pool_{s}_r{seed}',[R/'pool.py','--scene',s,'--seed',seed]
def h1(s,seed,cid):return f'h1_{s}_r{seed}_{cid}',[R/'h1.py','--scene',s,'--seed',seed,'--candidate',cid]

def main():
    p=load(R/'protocol.json');cases=[(s,seed) for s in p['scenes'] for seed in p['seeds']];first=cases[0]
    run([closed(*first,'A')],'PILOT_BASELINE');run([pool(*first)],'PILOT_POOL')
    meta=load(origin_dir(*first)/'catalog.json')
    run([h1(*first,'A')],'PILOT_H1_A')
    run([h1(*first,x['candidate_id']) for x in meta['representatives'] if x['candidate_id']!='A'],'PILOT_H1_OTHERS')
    dump(R/'pilot_passed.json',dict(case=first,baseline_tail_exact=True,origin_and_policy_exact=True,only_one_intervention=True,continuation_mppi_calls=0))
    run([closed(s,seed,'A') for s,seed in cases[1:]],'BASELINES')
    run([pool(s,seed) for s,seed in cases[1:]],'POOLS')
    h1jobs=[];unreachable=[]
    for s,seed in cases:
        directory=origin_dir(s,seed)
        if (directory/'unreachable.json').exists():unreachable.append(load(directory/'unreachable.json'));continue
        for x in load(directory/'catalog.json')['representatives']:
            cid=x['candidate_id']
            if not (R/'branches'/f'{s}_r{seed}'/cid/'episode.json').exists():h1jobs.append(h1(s,seed,cid))
    dump(R/'h1_expected.json',dict(cases=cases,unreachable=unreachable,remaining=len(h1jobs),pool_fixed=True))
    # Complete all H1 labels before decision; Native full episodes use only clean MAIN.
    run(h1jobs,'H1')
    dump(R/'h1_complete.json',dict(branches=len(list((R/'branches').glob('*/*/episode.json'))),unreachable=unreachable,cases=36))
    run([closed(s,seed,'B') for s,seed in cases],'NATIVE_CLOSED')
    dump(R/'native_closed_complete.json',dict(episodes=36))
    while not (R/'core_ready.json').exists():time.sleep(2)
    assert load(R/'core_ready.json')['tests_passed']
    run([closed(s,seed,'C') for s,seed in cases],'MODIFIED_CLOSED')
    dump(R/'closed_complete.json',dict(episodes=108,cases=36,maps=12,seeds=[1,2,3]))
    print('ALL_CLOSED108_COMPLETE',flush=True)
if __name__=='__main__':main()
