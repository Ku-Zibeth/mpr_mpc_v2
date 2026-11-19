"""Isolated synchronized full planner timings at replay-verified fixed real states.

Not a closed-loop effectiveness trial; no environment step is taken while timing.
"""
import argparse,time,random
from runtime import *
ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True);ap.add_argument('--method',required=True);a=ap.parse_args()
@torch.no_grad()
def main():
    _,_,env,agent,_=make_runtime(a.scene,1,a.method)
    from common.seed import set_seed
    rows=[]
    try:
        set_seed(a.scene+1);agent.planner.reset(seed=a.scene+1);obs=env.reset(seed=a.scene)
        trace=np.load(R/'development'/f'A_{a.scene}_r1/trace.npz');enabled=agent.planner.allow_mppi or a.method=='N';agent.planner.allow_mppi=False
        for i in range(100):
            assert np.array_equal(obs.cpu().numpy(),trace['observations'][i])
            action=agent.act(obs,t0=i==0,eval_mode=True).cpu().numpy();assert np.array_equal(action,trace['actions'][i])
            obs,_,done,_=env.step(action);assert not done
        assert np.array_equal(obs.cpu().numpy(),trace['observations'][100]);agent.planner.allow_mppi=enabled
        for repeat in range(35):
            set_seed(a.scene+1);agent.planner.local_mppi.reset(seed=a.scene+1);torch.cuda.synchronize();start=time.perf_counter()
            agent.act(obs,t0=False,eval_mode=True);torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
            if repeat>=5:rows.append(dict(configuration=a.method,scene=a.scene,replay_step=100,repeat=repeat-5,full_agent_act_ms=elapsed,
                **{k:float(v) for k,v in agent.last_plan.metrics.items() if k.endswith('_ms')}))
        directory=R/'isolated_latency';directory.mkdir(exist_ok=True)
        table(directory/f'{a.method}_{a.scene}.csv',rows)
        print('ISOLATED_LATENCY',a.method,a.scene,np.quantile([x['full_agent_act_ms'] for x in rows],[.5,.95,.99]),flush=True)
    finally:env.close()
if __name__=='__main__':main()
