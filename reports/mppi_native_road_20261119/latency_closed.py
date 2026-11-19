"""Sequential full closed-loop timing on predeclared first two development maps."""
from study import *
if __name__=='__main__':
    assert (R/'development_complete.json').exists()
    scenes=load(R/'protocol.json')['development_scenes'][:2];methods=['A','Tconditional']
    dump(R/'closed_latency_protocol.json',dict(scenes=scenes,methods=methods,workers=1,full_native_termination=True,not_new_independent_evidence=True,clock='mpr/total_planning_ms: actual coordinator planning wall time; env.step excluded; no outer collector checks in this clock'))
    for scene in scenes:
        for method in methods:
            with (R/f'closed_latency_{method}_{scene}.log').open('x') as out:
                subprocess.run([sys.executable,'-u',str(R/'closed.py'),'--split','latency','--scene',str(scene),'--seed','1','--method',method],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
            a=np.load(R/'development'/f'{method}_{scene}_r1/trace.npz');b=np.load(R/'latency'/f'{method}_{scene}_r1/trace.npz')
            assert all(np.array_equal(a[k],b[k]) for k in ['actions','observations','states','rewards','costs','done','terminated','truncated'])
            print('SEQUENTIAL_LATENCY_EXACT_REPRODUCTION',method,scene,flush=True)
    dump(R/'closed_latency_complete.json',dict(episodes=4,original_trace_reproduced_exactly=True,extra_independent_maps=0))
