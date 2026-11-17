"""Non-invasive phase profile on cached latent/actions, without a simulator."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import ROOT,R,load,dump,table,model_runtime
import numpy as np
import torch
from omegaconf import OmegaConf
from planning.local_mppi import LocalMPPI
from planning.timing import PhaseTimings


@torch.no_grad()
def main():
    assert not (R/'cached_phase_profile.json').exists()
    agent,evaluator=model_runtime(load(R/'experiment_manifest.json'))
    folder=next((ROOT/'reports/mppi_root_cause_20261117/catalog').iterdir());a=np.load(folder/'catalog.npz')
    z=torch.tensor(a['z0'],device='cuda');baseline=torch.tensor(a['actions'][0],device='cuda')
    cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi)
    cfg.update(iterations=4,selection_mode='weighted_mean',record_candidates=False,audit_candidates=False)
    planner=LocalMPPI(evaluator,cfg,horizon=10,action_dim=2,device='cuda')
    cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state();planner.reset(1)
    reference=planner.plan(z,baseline,eval_mode=True);aftercpu=torch.random.get_rng_state();aftergpu=torch.cuda.get_rng_state();afterprivate=planner.generator.get_state()
    phase=None;original=[]
    def install(obj,method,name):
        fn=getattr(obj,method);original.append((obj,method,fn))
        def wrapped(*args,**kwargs):
            label=name
            if name=='candidate_evaluation' and len(args[1])==2:label='final_candidate_evaluation_inclusive'
            with phase.measure(label):return fn(*args,**kwargs)
        setattr(obj,method,wrapped)
    for name in ['reward','next','termination']:install(agent.model,name,dict(reward='wm_reward',next='wm_dynamics',termination='wm_termination')[name])
    install(evaluator,'_terminal_prediction','q_evaluation')
    install(evaluator,'evaluate_mppi_candidates','candidate_evaluation')
    install(evaluator,'rescore_cached_candidates','final_elite_rescore_inclusive')
    install(planner,'_candidate_batch','mppi_sampling')
    install(planner,'_update_distribution','elite_update')
    rows=[]
    for i in range(36):
        if i==0:torch.random.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);planner.reset(1)
        phase=PhaseTimings('cuda');torch.cuda.synchronize();start=time.perf_counter()
        result=planner.plan(z,baseline,eval_mode=True);torch.cuda.synchronize();total=1000*(time.perf_counter()-start)
        values=phase.finish()
        if i==0:
            assert torch.equal(result.selected_sequence,reference.selected_sequence)
            assert torch.equal(aftercpu,torch.random.get_rng_state()) and torch.equal(aftergpu,torch.cuda.get_rng_state())
            assert torch.equal(afterprivate,planner.generator.get_state())
        if i>=6:rows.append(dict(total_cpu_wall_ms=total,**values))
    for obj,name,fn in reversed(original):setattr(obj,name,fn)
    summary=[]
    for key in rows[0]:
        v=np.array([r[key] for r in rows]);summary.append(dict(configuration='M4_cached_phase_profile',phase=key,
            source='same cachedorigin; per-plan accumulated PhaseTimings; nested phases not additive; event instrumentation overhead',samples=len(v),
            p50_ms=float(np.quantile(v,.5)),p95_ms=float(np.quantile(v,.95)),p99_ms=float(np.quantile(v,.99)),mean_ms=float(v.mean())))
    dump('cached_phase_profile.json',dict(origin=folder.name,rows=rows,summary=summary,
        action_and_global_private_RNG_identical=True,environment_runs=0,training_runs=0,
        scope='Frozen cached latent/actions only. Total is LocalMPPI, not fullLattice+SAC planner. Reward/dynamics/termination/Q are disjoint; inclusive eval stages overlap them.'))
    timings=load(R/'timing_statistics.json');timings['rows'].extend(summary);dump('timing_statistics.json',timings);table('timing_breakdown.csv',timings['rows'])
    print('cached phase profile; identical action/global/private RNG',flush=True)

if __name__=='__main__':main()
