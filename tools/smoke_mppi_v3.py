"""Fixed-state closed-loop smoke for the single simplified TD-MPC2 MPPI."""
import argparse,contextlib,io,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'logs/metadrive-risk/1/mpr_mpc_v2_1.0_wm200k_rl300k_20261103_195237/models/milestone_500k_full_mpr_mpc.pt')
    p.add_argument('--run',type=Path,default=ROOT/'logs/benchmark/mpr_mpc_v311_sac_vs_sac_mppi_fixed100_20261109_202823')
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--scenarios',type=int,nargs='+',default=[1714,559])
    p.add_argument('--snapshots',nargs='+',help='Optional exact archived-action prefixes: scenario:before_step')
    p.add_argument('--stack',choices=['world_model_residual_rl','full_mpr_mpc'],default='full_mpr_mpc')
    a=p.parse_args()
    if a.output.exists() or not 0<a.steps<=200:p.error('Require fresh output and 1..200 steps.')
    snapshots=dict(tuple(map(int,x.split(':'))) for x in a.snapshots) if a.snapshots else {s:1 for s in a.scenarios}
    records={s:[] for s in snapshots}
    if a.snapshots:
        for line in next((a.run/'steps').glob('*_full_mpr_mpc_seed1.jsonl')).open():
            row=json.loads(line);s=row['scenario']
            if s in snapshots and row['episode_step']<snapshots[s]:records[s].append(row)
            if all(len(records[s])==n-1 for s,n in snapshots.items()):break
    raw=OmegaConf.load(a.run/'config_D_lattice_sac_mppi_seed1.yaml')
    raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
    raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8
    _,cfg=suite._make_cfg(raw,a.checkpoint.parent.parent.name,a.checkpoint,list(snapshots));cfg.residual_rl['buffer_size']=8
    torch.set_float32_matmul_precision('high');set_seed(cfg.seed);env=make_env(cfg);cases=[]
    checksum=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    try:
        with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);loaded=agent.load(a.checkpoint)
        agent.eval();suite._apply_evaluation_stack(agent.planner,a.stack,loaded)
        agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
        for s,before in snapshots.items():
            set_seed(int(cfg.seed)+s);agent.planner.reset(seed=int(cfg.seed)+s);obs=env.reset(seed=s)
            info={};replay_error=0.
            for old in records[s]:
                replay_error=max(replay_error,abs(np.linalg.norm(env.unwrapped_metadrive.agent.velocity)-old['mpr/actual_speed_mps']))
                obs,_,done,info=env.step(np.array(old['executed_action'],np.float32))
                if done:raise RuntimeError('Archived prefix terminated early.')
            if replay_error>1e-4:raise RuntimeError(f'Archived prefix mismatch: {replay_error}')
            route0=float(info.get('route_completion',0.));rows=[]
            for i in range(a.steps):
                action=agent.act(obs,t0=(i==0 and before==1),eval_mode=True);result=agent.last_plan
                if result.stage.use_mppi:
                    seq=result.debug['selected_mppi_actions'];base=result.refined_actions.cpu()
                    assert seq.shape==base.shape==(cfg.horizon+1,cfg.action_dim)
                    assert (seq.abs()<=1).all() and ((seq-base).abs()<=agent.planner.local_mppi.max_delta.cpu()+1e-7).all()
                    assert 'safety_certificate' not in result.debug
                    assert not result.safety_failure
                    if not result.metrics['mppi/invalid_prediction']:
                        assert result.metrics['mpr/selected_wm_value']>=result.metrics['mpr/baseline_wm_value']
                        assert result.metrics['mppi/iterations_completed']==cfg.mpr_mpc['mppi']['iterations']
                obs,reward,done,info=env.step(action)
                rows.append(dict(step=i+1,action=action.tolist(),reward=float(reward),route=float(info.get('route_completion',0.)),
                    speed=float(np.linalg.norm(env.unwrapped_metadrive.agent.velocity)),
                    crash=any(bool(info.get(k,False)) for k in ('crash','crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk')),
                    offroad=bool(info.get('out_of_road',False)),safety_failure=bool(info.get('safety_failure',False)),
                    mode=result.planning_mode,metrics={k:float(v) for k,v in result.metrics.items() if isinstance(v,(float,int,np.number))}))
                if done:break
            case=dict(scenario=s,snapshot_before_step=before,replay_max_speed_error=replay_error,records=rows,
                summary=dict(steps=len(rows),route_gain=float(info.get('route_completion',0.))-route0,
                    mean_speed=float(np.mean([r['speed'] for r in rows])),crash=any(r['crash'] for r in rows),
                    offroad=any(r['offroad'] for r in rows),safety_failure=any(r['safety_failure'] for r in rows),
                    baseline_selected=float(np.mean([r['mode']=='baseline' for r in rows])),
                    planning_mean_ms=float(np.mean([r['metrics']['mpr/total_planning_ms'] for r in rows])),
                    mean_score_gain=float(np.mean([r['metrics'].get('mppi/score_gain',0) for r in rows]))))
            cases.append(case);print(s,case['summary'],flush=True)
    finally:env.close()
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==checksum
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(checkpoint=str(a.checkpoint),checkpoint_sha256=checksum,stack=a.stack,results=cases,
        note='Bounded smoke only; task success is evaluated separately over complete episodes.'),indent=2))


if __name__=='__main__':main()
