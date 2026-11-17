"""Frozen original runtime for offline replay; no learner update or new score."""
import contextlib,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'))
from probe import dump,snapshot
import numpy as np,torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite


def make(scenario):
 p=json.loads((R/'protocol.json').read_text());ck=Path(p['checkpoint']);raw=OmegaConf.load(p['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi;raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8;raw.metadrive.simulator.horizon=1000
 _,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[scenario]);torch.set_num_threads(1);torch.set_float32_matmul_precision('high');set_seed(1);env=make_env(cfg)
 with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);state=agent.load(ck)
 agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',state);agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step));return p,cfg,env,agent


def at_origin(env,agent,scenario,before):
 seed=1+scenario;set_seed(seed);agent.planner.reset(seed=seed);obs=env.reset(seed=scenario)
 for i in range(before-1):
  action=agent.act(obs,t0=(i==0),eval_mode=True);obs,_,done,info=env.step(action)
  if done:raise RuntimeError(f'Prespecified origin unreachable: {scenario}:{before}, step{i+1}')
 agent.act(obs,t0=False,eval_mode=True);return obs,agent.planner.last_context,agent.last_plan.refined_actions.clone()


def physical(env):
 v=env.unwrapped_metadrive.agent
 return np.array([*map(float,v.position),float(v.heading_theta),float(np.linalg.norm(v.velocity))],np.float64)


def actor_audit(agent,state):
 x=state.reshape(1,-1);actor=agent.residual_sac.actor;base=actor(x,deterministic=True)[0];deltas={}
 for name,(lo,hi) in dict(path=(512,524),wm_return=(524,525),feasibility=(525,527),risk=(527,535)).items():
  changed=x.clone();changed[:,lo:hi]=0.;out=actor(changed,deterministic=True)[0];deltas[name]=(out-base).cpu().numpy().reshape(-1)
 return base.cpu().numpy().reshape(-1),deltas
