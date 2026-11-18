"""Real same-state coarse535/SAC parity and converter replay, no D episode."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import torch
from study_io import R,load,dump,make_runtime
from tools.first_action_study import snapshot


@torch.no_grad()
def main():
    assert 'passed' in (R/'tests_alignment.log').read_text()
    p=load(R/'protocol.json');scene=p['scenarios'][0];seed=1
    parsed,cfg,env,agent=make_runtime(scene,seed,'D',record=True)
    from common.seed import set_seed
    try:
        set_seed(scene+seed);obs=env.reset(seed=scene);agent.planner.allow_mppi=False
        for step in range(29):
            action=agent.act(obs,t0=step==0,eval_mode=True);obs,_,done,_=env.step(action);assert not done
        planner=agent.planner;planner.allow_mppi=True;vehicle=env.unwrapped_metadrive.agent
        planner.mppi_cfg['cartesian_virtual_speed']=False
        c0=planner.prepare(obs,vehicle)
        cpu=torch.random.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();private=planner.local_mppi.generator.get_state().clone()
        before=planner.refine_and_plan(c0,vehicle,eval_mode=True,global_step=agent.global_env_step)
        planner.mppi_cfg['cartesian_virtual_speed']=True
        c1=planner.prepare(obs,vehicle)
        torch.random.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);planner.local_mppi.generator.set_state(private)
        after=planner.refine_and_plan(c1,vehicle,eval_mode=True,global_step=agent.global_env_step)
        checks=dict(coarse535=torch.equal(c0.residual_state,c1.residual_state),coarse_actions=torch.equal(c0.coarse_actions,c1.coarse_actions),
            latent=torch.equal(c0.latent,c1.latent),requested_SAC=torch.equal(before.requested_normalized_residual,after.requested_normalized_residual),
            refined_parameters=np.array_equal(before.refined_parameters,after.refined_parameters),
            baseline_first_action=torch.equal(before.refined_actions[0],after.refined_actions[0]))
        assert all(checks.values()),checks
        np.savez_compressed(R/'D_preflight_arrays.npz',coarse535=c0.residual_state.cpu().numpy(),
            legacy_refined_actions=before.refined_actions.cpu().numpy(),cartesian_refined_actions=after.refined_actions.cpu().numpy(),
            selected_C=before.action.cpu().numpy(),selected_D=after.action.cpu().numpy())
        dump('D_preflight.json',dict(passed=True,checks=checks,scene=scene,pre_action_step=30,
            tail_control_max_difference=float((before.refined_actions[1:]-after.refined_actions[1:]).abs().max()),
            same_initial_state_hashes=snapshot(env,obs)['hashes'],prefix_steps=29,executed_D_actions=0,
            note='One physical Baseline prefix used only to rebuild a real state. Probe C/D actions not executed; native episodes run separately.'))
        print('D_PREFLIGHT_PASSED',checks,flush=True)
    finally:env.close()


if __name__=='__main__':main()
