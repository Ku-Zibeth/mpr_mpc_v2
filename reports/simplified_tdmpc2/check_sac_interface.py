"""Snapshot the unchanged 535-D SAC input/output on fixed fresh scenes."""
import argparse,contextlib,io,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
run=ROOT/'logs/metadrive-risk/1/mpr_mpc_v2_1.0_wm200k_rl300k_20261103_195237';ck=run/'models/milestone_500k_full_mpr_mpc.pt'
raw=OmegaConf.load(ROOT/'logs/benchmark/mpr_mpc_v311_sac_vs_sac_mppi_fixed100_20261109_202823/config_B_lattice_sac_seed1.yaml')
raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_video=raw.save_agent=False;raw.residual_rl.buffer_size=8
_,cfg=suite._make_cfg(raw,run.name,ck,[1714,559]);cfg.residual_rl['buffer_size']=8
torch.set_float32_matmul_precision('high');set_seed(cfg.seed);env=make_env(cfg)
try:
 with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);state=agent.load(ck)
 agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',state)
 agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step));data=[]
 for scenario in (1714,559):
  set_seed(1+scenario);obs=env.reset(seed=scenario);action=agent.act(obs,t0=True,eval_mode=True)
  ctx=agent.planner.last_context;r=agent.last_plan
  assert ctx.residual_state.numel()==535
  data.append(dict(scenario=scenario,state=ctx.residual_state.cpu().flatten().tolist(),
   residual=r.requested_normalized_residual.cpu().tolist(),refined_actions=r.refined_actions.cpu().tolist(),action=action.cpu().tolist()))
 a.output.write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),cases=data),indent=2));print(a.output)
finally:env.close()
