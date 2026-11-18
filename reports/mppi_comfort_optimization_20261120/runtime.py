"""Frozen original Lattice/SAC, identical environment; explicit comfort toggle only."""
import contextlib,io
from comfort_study import *

def make_runtime(scene,seed,group):
    from mpr_mpc._bootstrap import bootstrap
    bootstrap()
    from omegaconf import OmegaConf
    from common.seed import set_seed
    from env import make_env
    import evaluate_model_suite as suite
    p=load(R/'protocol.json');checkpoint=Path(p['checkpoint'])
    raw=OmegaConf.load(p['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
    if group=='C':
        for key,value in load(R/'comfort_config.json').items():raw.mpr_mpc.mppi[key]=value
    raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_agent=raw.save_video=False
    raw.residual_rl.buffer_size=8;raw.seed=seed
    parsed,cfg=suite._make_cfg(raw,checkpoint.parent.parent.name,checkpoint,[scene])
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');set_seed(1)
    env=make_env(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        agent=suite._make_agent(cfg,env);state=agent.load(checkpoint)
    agent.eval();suite._apply_evaluation_stack(agent.planner,'full_mpr_mpc' if group=='C' else 'world_model_residual_rl',state)
    agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
    assert agent.planner.residual_state_dim==535 and agent.planner.allow_mppi==(group=='C')
    assert float(agent.tdmpc_agent.discount)==.995 and agent.planner.control_dt==.1
    local=agent.planner.local_mppi
    assert local.num_samples==64 and local.num_elites==8 and local.iterations==4 and local.cost_weight==0
    if group=='C':
        assert local.objective_mode=='comfort'
        def forbidden(*args,**kwargs):raise AssertionError('Candidate WM/Q must never run in comfort mode')
        agent.planner.evaluator.evaluate_mppi_candidates=forbidden
    return parsed,cfg,env,agent
