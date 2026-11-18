"""Strict phase gates, independent subprocesses, no retries or outcome-based sample changes."""
import os,subprocess,sys,time
from study import R,ROOT,load,dump,origin_dir

def run(jobs,label,workers=4):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    active=[];completed=[];queue=list(jobs);logs=R/'logs';logs.mkdir(exist_ok=True)
    try:
        while queue or active:
            while queue and len(active)<workers:
                name,args=queue.pop(0)
                stream=(logs/f'{name}.log').open('x')
                child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,child,stream))
            for name,child,stream in active[:]:
                status=child.poll()
                if status is None:continue
                active.remove((name,child,stream));stream.close()
                assert status==0,('phase stopped',label,name,status)
                completed.append(name);print(label,name,len(completed),'/',len(jobs),flush=True)
            if active:time.sleep(1)
    finally:
        for _,child,stream in active:
            if child.poll() is None:child.terminate()
            child.wait();stream.close()
    return completed

def branch_job(scene,step,cid):
    return f'branch_{scene}_{step}_{cid}',[str(R/'branch.py'),'--scene',str(scene),'--step',str(step),'--candidate',cid]

def main():
    p=load(R/'protocol.json');scene,step=p['pilot_origin']
    a0=load(R/'branches'/f'{scene}_{step}'/'A0'/'episode.json')
    assert a0['baseline_full_tail_matched']>=10 and a0['exact_origin'] and a0['exact_policy_history']
    assert '41 passed' in (R/'tests_v2.log').read_text()
    dump(R/'phase1_passed.json',dict(pilot=[scene,step],whole_baseline_tail_exact=a0['baseline_full_tail_matched'],tests_passed=41))
    done=run([branch_job(scene,step,cid) for cid in ['A1','A3','A4','A5']],'PILOT')
    for cid in ['A1','A3','A4','A5']:
        row=load(R/'branches'/f'{scene}_{step}'/cid/'episode.json')
        assert row['intervention_steps']==1 and row['continuation_mppi_calls']==0 and row['exact_origin'] and row['exact_policy_history'] and row['extended_environment_exact']
    dump(R/'phase2_passed.json',dict(pilot=[scene,step],branches=done,all_checks_passed=True))
    origins=[(s,t) for s in p['scenes'] for t in p['steps']]
    prep=[(f'prepare_{s}_{t}',[str(R/'prepare.py'),'--scene',str(s),'--step',str(t)]) for s,t in origins if (s,t)!=(scene,step)]
    run(prep,'PREPARE')
    run([branch_job(s,t,'A0') for s,t in origins if (s,t)!=(scene,step)],'BASELINE')
    dump(R/'all_baselines_passed.json',dict(origins=origins,full_tail_exact=True))
    jobs=[]
    for s,t in origins:
        catalog=load(origin_dir(s,t)/'catalog.json')
        for x in catalog['representatives']:
            cid=x['candidate_id']
            if not (R/'branches'/f'{s}_{t}'/cid/'episode.json').exists():jobs.append(branch_job(s,t,cid))
    rest=run(jobs,'H1')
    expected=sum(len(load(origin_dir(s,t)/'catalog.json')['representatives']) for s,t in origins)
    actual=list((R/'branches').glob('*/*/episode.json'));assert len(actual)==expected
    dump(R/'collection_complete.json',dict(origins=16,maps=8,branches=len(actual),remaining_completed=rest,
        native_done=True,intervention_horizon=1,formal_h1_ranking='NOT ESTIMABLE'))
    print('ALL_H1_BRANCHES_COMPLETE',expected,flush=True)
if __name__=='__main__':main()
