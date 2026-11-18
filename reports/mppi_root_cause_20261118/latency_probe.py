"""Single-worker full-agent latency at identical frozen physical states; no probe action execution."""
import argparse,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--group',choices=['B','C','D'],required=True);p.add_argument('--scene',type=int,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root if a.group=='D' else root.parent/'mpr_mpc_v2'))
import numpy as np
import torch
from study_io import R,load,dump,table,make_runtime
from tools.first_action_study import snapshot

@torch.no_grad()
def main():
    assert (R/'D_collection_complete.json').exists(),'Do not contend with native collection'
    parsed,cfg,env,agent=make_runtime(a.scene,1,a.group,record=False)
    from common.seed import set_seed
    try:
        set_seed(a.scene+1);agent.planner.reset(seed=a.scene+1);obs=env.reset(seed=a.scene)
        agent.planner.allow_mppi=False
        for i in range(29):
            action=agent.act(obs,t0=i==0,eval_mode=True);obs,_,done,_=env.step(action);assert not done
        origin=snapshot(env,obs);agent.planner.allow_mppi=a.group!='B'
        phases={};round_index=[-1];evaluation_index=[-1]
        def measured(obj,name,key,sampling=False):
            old=getattr(obj,name)
            def call(*args,**kwargs):
                if sampling:round_index[0]+=1
                if name=='evaluate_mppi_candidates':evaluation_index[0]+=1
                begin=time.perf_counter();result=old(*args,**kwargs)
                label=key.format(iteration=round_index[0],batch=evaluation_index[0]);phases[label]=phases.get(label,0.)+1000*(time.perf_counter()-begin)
                return result
            setattr(obj,name,call)
        planner=agent.planner
        measured(planner.structured,'select_coarse','lattice_cpu_wall_ms')
        measured(planner.action_adapter,'path_to_actions','adapter_cpu_wall_ms')
        measured(planner.local_mppi,'_candidate_batch','iteration_{iteration}_sampling_cpu_wall_ms',True)
        measured(planner.local_mppi,'_update_distribution','iteration_{iteration}_update_cpu_wall_ms')
        measured(planner.evaluator,'evaluate_mppi_candidates','evaluation_batch_{batch}_cpu_wall_ms')
        measured(planner.evaluator,'_terminal_prediction','terminal_prediction_cpu_wall_ms')
        rows=[]
        for i in range(55):
            phases.clear();round_index[0]=-1;evaluation_index[0]=-1
            torch.cuda.synchronize();t=time.perf_counter();agent.act(obs,t0=False,eval_mode=True);torch.cuda.synchronize()
            elapsed=1000*(time.perf_counter()-t)
            if i>=5:
                rows.append(dict(configuration=a.group,scene=a.scene,repeat_seed=1,probe=i-5,
                    total_wall_ms=elapsed,scope='isolated full agent.act; same physical state; no candidate recording',
                    **phases,**{k:float(v) for k,v in agent.last_plan.metrics.items() if 'ms' in k}))
        final=snapshot(env,obs)
        assert all(final['hashes'][k]==origin['hashes'][k] for k in ['observation','vehicle','traffic']),'Probe changed physical state'
        assert all(final['rng_component_hashes'][k]==origin['rng_component_hashes'][k] for k in ['engine','traffic_manager']),'Probe changed simulator RNG'
        # Torch/planner RNG advances normally; simulator snapshot RNG excludes these in physical hashes below.
        table(f'probe_{a.group}_{a.scene}.csv',rows)
        dump(f'probe_{a.group}_{a.scene}.json',dict(scene=a.scene,configuration=a.group,physical_state=origin,
             prefix_steps=29,warmup=5,measured=50,probe_actions_executed=0,p95_ms=float(np.quantile([r['total_wall_ms'] for r in rows],.95))))
        print('PROBE_DONE',a.group,a.scene,flush=True)
    finally:env.close()
if __name__=='__main__':main()
