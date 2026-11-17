"""Real-checkpoint GPU checks: shared members, cached scoring, frozen SAC features."""
import contextlib,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump
import numpy as np,torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite
from planning.local_mppi import LocalMPPI


@torch.no_grad()
def main():
 p=json.loads((R/'protocol.json').read_text());ck=Path(p['checkpoint']);scene=p['development_scenarios'][0];raw=OmegaConf.load(p['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi;raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8
 _,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[scene]);torch.set_num_threads(1);torch.set_float32_matmul_precision('high');set_seed(1);env=make_env(cfg)
 try:
  with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);state=agent.load(ck)
  agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',state);agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step));set_seed(scene+1);agent.planner.reset(seed=scene+1);obs=env.reset(seed=scene);agent.act(obs,t0=True,eval_mode=True);context=agent.planner.last_context;baseline=agent.last_plan.refined_actions;old=context.residual_state.clone();wm=agent.planner.evaluator.evaluate_trajectory_consequence(context.latent,context.coarse_actions[None]).features.clone();rows=[]
  for terminal in ['candidate','policy_prior']:
   for selection in ['weighted_mean','best_final_elite']:
    options=dict(cfg.mpr_mpc['mppi']);options.update(iterations=2,terminal_action=terminal,selection_mode=selection,audit_candidates=False,record_candidates=True);local=LocalMPPI(agent.planner.evaluator,options,horizon=10,action_dim=2,device=agent.device);local.reset(scene+1);set_seed(scene+1);result=local.plan(context.latent,baseline,eval_mode=True);record=result.selection_candidates;direct=agent.planner.evaluator.evaluate_mppi_candidates(context.latent,record['actions'],compute_cost=False,terminal_action=terminal,q_aggregation='fixed_pair',q_indices=record['q_member_indices']);error=(direct.total_value-record['values']).abs();assert float(error.max())<.01,float(error.max());assert torch.equal(result.action,result.selected_sequence[0]);assert torch.equal(context.residual_state,old)
    if terminal=='policy_prior':assert (result.final_std[-1]==0).all() and torch.equal(result.final_mean[-1],baseline[-1])
    rows.append(dict(terminal_action=terminal,selection_mode=selection,shared_Q=record['q_member_indices'],values=record['values'].cpu().tolist(),direct_three_candidate_values=direct.total_value.cpu().tolist(),maximum_cached_vs_direct_error=float(error.max()),source=record['selected_source'],frozen_SAC535_max_error=float((context.residual_state-old).abs().max())))
  after=agent.planner.evaluator.evaluate_trajectory_consequence(context.latent,context.coarse_actions[None]).features;assert torch.equal(wm,after);dump(R/'gpu_regression.json',dict(rows=rows,SAC535_unchanged=True,coarse_SAC_WM_features_unchanged=True,note='Direct3-batch vs cached1/2-batch GPU comparison has small TF32 rounding; .01 absolute score tolerance for diagnostic only, never a selection margin. All sampling/online comparisons unchanged.'))
 finally:env.close()
if __name__=='__main__':main()
