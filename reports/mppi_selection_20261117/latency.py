"""Isolated complete planning on two identical live states; not driving outcomes."""
import contextlib,io,json,sys,time
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
sys.path.insert(0,str(R))
from closed_loop import method


@torch.no_grad()
def main():
 p=json.loads((R/'protocol.json').read_text());ck=Path(p['checkpoint']);rows=[];states=[];torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
 for scene in p['development_scenarios'][:2]:
  raw=OmegaConf.load(p['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi;raw.mpr_mpc.mppi.update(dict(selection_mode='weighted_mean',audit_candidates=False,record_candidates=False,record_cost=False,cost_weight=0.));raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8;raw.metadrive.simulator.horizon=1000
  _,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[scene]);set_seed(1);env=make_env(cfg)
  try:
   with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);state=agent.load(ck)
   agent.eval();agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step));suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl',state);set_seed(1+scene);agent.planner.reset(seed=1+scene);obs=env.reset(seed=scene)
   for step in range(30):
    action=agent.act(obs,t0=(step==0),eval_mode=True);obs,_,done,info=env.step(action);assert not done
   v=env.unwrapped_metadrive.agent;states.append(dict(scenario=scene,position=v.position.tolist(),heading=float(v.heading_theta),speed=float(v.speed),observation=obs.cpu().tolist()))
   for group in ['B','M1','M2','M4','P','E','PE']:
    option=method(group,p);op=dict(cfg.mpr_mpc['mppi']);op.update({k:val for k,val in option.items() if k!='mppi'});agent.planner.local_mppi=LocalMPPI(agent.planner.evaluator,op,horizon=10,action_dim=2,device=agent.device);agent.planner.mppi_cfg=op;agent.planner.allow_mppi=option['mppi'];set_seed(1+scene);agent.planner.reset(seed=1+scene)
    for _ in range(5):agent.act(obs,t0=False,eval_mode=True)
    for i in range(50):
     torch.cuda.synchronize();begin=time.perf_counter();agent.act(obs,t0=False,eval_mode=True);torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-begin);rows.append(dict(scenario=scene,configuration=group,call=i,planning_time_ms=elapsed))
  finally:env.close()
 summary={}
 for group in ['B','M1','M2','M4','P','E','PE']:
  x=np.array([r['planning_time_ms'] for r in rows if r['configuration']==group]);summary[group]=dict(calls=len(x),mean_ms=float(x.mean()),p50_ms=float(np.median(x)),p95_ms=float(np.quantile(x,.95)),max_ms=float(x.max()),mean_and_p95_within_control_period=bool(x.mean()<=100 and np.quantile(x,.95)<=100))
 dump(R/'isolated_latency.json',dict(summary=summary,rows=rows,states=states,measurement='One GPU worker after experiments; complete Lattice+SAC+MPPI+action construction, synchronized. Same two live states,50calls/state plus5warmups, no candidate recorder, no control executed during timing. Fixed-state reference only; does not certify every episode state.'))
 print(summary,flush=True)
if __name__=='__main__':main()
