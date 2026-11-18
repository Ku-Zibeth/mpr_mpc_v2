"""Real frozen-model flag wiring at one reconstructed state; probe outputs never executed."""
import os
from pathlib import Path
os.environ['MPR_CODE_ROOT']=str(Path(__file__).resolve().parents[2])
from study import *

@torch.no_grad()
def main():
    p=load(R/'protocol.json');scene=p['scenes'][0];seed=1
    parsed,cfg,env,agent=make_runtime(scene,seed,'modified',record=False)
    try:
        local=agent.planner.local_mppi
        assert local.final_selection_policy=='baseline_guard' and not local.replacement_supported
        agent.planner.allow_mppi=False
        obs,baseline,_,before,policy,_=restore_origin(env,agent,scene,seed)
        agent.planner.allow_mppi=True;rng=get_rng();private=local.generator.get_state().clone()
        protected=agent.act(obs,t0=False,eval_mode=True);plan=agent.last_plan
        assert torch.equal(protected.cpu(),baseline.cpu())
        assert int(plan.metrics['mppi/iterations_completed'])==4 and int(plan.metrics['mppi/wm_evaluated_candidates'])==262
        assert digest(agent.planner.last_context.residual_state)==policy['hashes']['residual535']
        assert plan.metrics['mppi/replacement_supported']==0
        set_rng(rng);local.generator.set_state(private);local.replacement_supported=True;local.final_selection_margin=0.
        permissive=agent.act(obs,t0=False,eval_mode=True)
        assert not torch.equal(permissive,protected),'Selected test state must exercise an actual proposed replacement'
        assert snapshot(env,obs)['hashes']['vehicle']==before['hashes']['vehicle']
        dump(R/'real_guard_probe.json',dict(passed=True,scene=scene,seed=seed,origin_step=50,
            frozen_options_reached_controller=True,unsupported_returns_original_sac=True,coarse535_unchanged=True,
            full_search_kept=[64,8,4],evaluated_candidates=262,
            supported_zero_margin_action_delta=float((permissive-protected).norm()),probe_actions_executed=0,
            note='Positive-support switch is exercised only in this isolated no-execution test. Experimental decision remains unsupported.'))
        print('REAL_GUARD_PROBE_PASSED',flush=True)
    finally:env.close()
if __name__=='__main__':main()
