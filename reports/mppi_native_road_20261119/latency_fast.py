"""Performance-only switch; require exact complete traces before accepting timing."""
from study import *
if __name__=='__main__':
    for scene in load(R/'protocol.json')['development_scenes'][:2]:
        with (R/f'closed_latency_Tfast_{scene}.log').open('x') as out:
            subprocess.run([sys.executable,'-u',str(R/'closed.py'),'--split','latency','--scene',str(scene),'--seed','1','--method','Tfast'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
        a=np.load(R/'latency'/f'Tconditional_{scene}_r1/trace.npz');b=np.load(R/'latency'/f'Tfast_{scene}_r1/trace.npz')
        assert all(np.array_equal(a[k],b[k]) for k in ['actions','observations','states','rewards','costs','done','terminated','truncated']),scene
        print('FAST_FULL_TRACE_EXACT',scene,flush=True)
    dump(R/'fast_full_trace_parity.json',dict(episodes=2,all_actions_observations_states_rewards_costs_termination_exact=True,extra_independent_maps=0))
