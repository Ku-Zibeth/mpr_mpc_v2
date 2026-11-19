"""Frozen policy, independent traffic repetitions, explicit tracking variant only."""
from study import *

def make_runtime(scene,seed,method):
    from mpr_mpc._bootstrap import bootstrap
    bootstrap()
    from omegaconf import OmegaConf
    from common.seed import set_seed
    from env import make_env
    import evaluate_model_suite as suite
    p=load(R/'protocol.json');ck=Path(p['checkpoint']);raw=OmegaConf.load(p['source_environment_config'])
    raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
    tracked=method.startswith('T') or method=='C'
    if tracked:
        tag=load(R/'selection.json')['selected'] if method=='C' else method
        variants=load(R/'variant_configs.json')
        if (R/'corrected_variant_configs.json').exists():variants.update(load(R/'corrected_variant_configs.json'))
        if (R/'resolution_variant_config.json').exists():variants.update(load(R/'resolution_variant_config.json'))
        for k,v in variants[tag].items():raw.mpr_mpc.mppi[k]=v
    raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8;raw.seed=seed
    parsed,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[scene])
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');set_seed(1);env=make_env(cfg)
    with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);state=agent.load(ck)
    agent.eval();suite._apply_evaluation_stack(agent.planner,'full_mpr_mpc' if tracked else 'world_model_residual_rl',state)
    agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
    assert agent.planner.residual_state_dim==535 and agent.planner.allow_mppi==tracked
    assert float(agent.tdmpc_agent.discount)==.995 and agent.planner.control_dt==.1
    local=agent.planner.local_mppi;assert (local.num_samples,local.num_elites,local.iterations)==(64,8,4)
    if tracked:
        assert local.objective_mode=='tracking' and str(local.device)=='cpu'
        def forbidden(*args,**kwargs):raise AssertionError('Tracking candidates must not use learned WM/Q scores')
        agent.planner.evaluator.evaluate_mppi_candidates=forbidden
    # Initialize engine identically, then seed only the native traffic manager
    # immediately before native traffic reset. Map/agent/reward/termination stay intact.
    env.reset(seed=scene);sim=env.unwrapped_metadrive;mgr=sim.engine.traffic_manager;original=mgr.reset
    assert not mgr.random_traffic
    traffic_seed=int(scene*1009+seed)
    def reset_traffic():
        mgr.seed(traffic_seed)
        return original()
    mgr.reset=reset_traffic
    return parsed,cfg,env,agent,traffic_seed
