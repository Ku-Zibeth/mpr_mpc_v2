"""Isolated FULL planner and LocalMPPI timings at identical source B physical states."""
import argparse,os,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['A','B','C'],required=True);ap.add_argument('--scene',type=int,required=True);a=ap.parse_args()
root=Path(__file__).resolve().parents[2];os.environ['MPR_CODE_ROOT']=str(root if a.group=='C' else root.parent/'mpr_mpc_v2')
from study import *
@torch.no_grad()
def main():
    parsed,cfg,env,agent=make_runtime(a.scene,1,'modified' if a.group=='C' else 'baseline',record=False)
    try:
        agent.planner.allow_mppi=False
        obs,_,_,state,_,_=restore_origin(env,agent,a.scene,1)
        agent.planner.allow_mppi=a.group!='A';local=[0.];old=agent.planner.local_mppi.plan
        def measured(*args,**kwargs):
            torch.cuda.synchronize();t=time.perf_counter();result=old(*args,**kwargs);torch.cuda.synchronize();local[0]=1000*(time.perf_counter()-t);return result
        agent.planner.local_mppi.plan=measured;rows=[]
        for i in range(55):
            local[0]=0.;torch.cuda.synchronize();t=time.perf_counter();agent.act(obs,t0=False,eval_mode=True);torch.cuda.synchronize();total=1000*(time.perf_counter()-t)
            if i>=5:rows.append(dict(configuration=a.group,scene=a.scene,seed=1,call=i-5,planner_ms=total,local_mppi_ms=local[0],
                **{k:float(v) for k,v in agent.last_plan.metrics.items() if k.endswith('_ms')}))
        end=snapshot(env,obs)
        assert all(end['hashes'][k]==state['hashes'][k] for k in ['observation','vehicle','traffic'])
        assert all(end['rng_component_hashes'][k]==state['rng_component_hashes'][k] for k in ['engine','traffic_manager'])
        table(R/f'probe_{a.group}_{a.scene}.csv',rows)
        dump(R/f'probe_{a.group}_{a.scene}.json',dict(initial_hashes=state['hashes'],warmup=5,calls=50,executed_probe_actions=0,
            scope='isolated full agent.act and LocalMPPI; no env.step. Complete cycle measured in native episodes separately.'))
        print('ISOLATED_PROBE_DONE',a.group,a.scene,flush=True)
    finally:env.close()
if __name__=='__main__':main()
