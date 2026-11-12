"""Bounded same-action-prefix replay; records failed certification without executing it."""
import argparse, contextlib, hashlib, io, json, sys, subprocess
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
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--scenario',type=int)
    p.add_argument('--fresh',action='store_true')
    a=p.parse_args()
    if a.output.exists() or not 1<=a.steps<=120:p.error('Fresh output and 1..120 steps required.')
    original=json.loads((ROOT/'reports/sac_refinement/paired_run.json').read_text())
    simplified=json.loads((ROOT/'reports/simplified_tdmpc2/paired_run.json').read_text())
    checkpoint=Path(simplified['checkpoint'])
    failures={r['scenario']:r['episode_step'] for r in json.loads((ROOT/'reports/sac_refinement/safety_failure_diagnosis.json').read_text())['cases']}
    origins={559:600,1714:250,**{s:failures[s] for s in (694,420,1084,808,822,1196,1534,1109,1960,408,1939)}}
    if a.scenario is None:
        directory=a.output.with_suffix('.parts');directory.mkdir()
        cases=[]
        for scenario in origins:
            child=directory/f'{scenario}.json'
            log=directory/f'{scenario}.log'
            with log.open('w') as stream:
                status=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(child),
                    '--steps',str(a.steps),'--scenario',str(scenario)],stdout=stream,stderr=subprocess.STDOUT)
            if status.returncode:
                failed_reason=log.read_text()[-2000:]
                child=directory/f'{scenario}_fresh.json'
                with (directory/f'{scenario}_fresh.log').open('w') as stream:
                    retry=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(child),
                        '--steps',str(a.steps),'--scenario',str(scenario),'--fresh'],stdout=stream,stderr=subprocess.STDOUT)
                if retry.returncode:
                    cases.append(dict(scenario=scenario,replay_available=False,reason=failed_reason))
                else:
                    rows=json.loads(child.read_text())['cases']
                    for row in rows:row['historical_replay_unavailable']=failed_reason
                    cases+=rows
            else:
                cases+=json.loads(child.read_text())['cases']
            print(scenario,'completed',status.returncode,flush=True)
        a.output.write_text(json.dumps(dict(cases=cases,requested_closed_loop_steps=a.steps,
            fixed_state_acceptance_passed=all(c.get('physical_steps')==a.steps and not any(r['crash'] or r['offroad'] or r['safety_failure'] for r in c.get('records',[])) for c in cases),
            checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            scope='Fresh process per scenario; exact prefix tolerance remains 1e-4. Compact original failure traces are explicitly labelled as a different V311 regression state. Unavailable replay is not treated as a passing test.'),indent=2))
        return
    origins={a.scenario:origins[a.scenario]}
    archived=ROOT/'logs/benchmark/mpr_mpc_v311_sac_vs_sac_mppi_fixed100_20261109_202823'
    streams=list((Path(original['output'])).glob('part*/steps/*_full_mpr_mpc_seed1.jsonl'))
    records={s:[] for s in origins}; old_available={s:[] for s in origins}; provenance={}
    for source in streams:
        for line in source.open():
            row=json.loads(line);s=row['scenario']
            if s in old_available and row['episode_step']<origins[s] and row.get('action_executed',True):old_available[s].append(row)
    full={s:[] for s in origins}
    for source in (archived/'steps').glob('*_full_mpr_mpc_seed1.jsonl'):
        for line in source.open():
            row=json.loads(line)
            if row['scenario'] in full:full[row['scenario']].append(row)
    for s in origins:
        if s not in (559,1714) and len(old_available[s])==origins[s]-1:
            records[s]=sorted(old_available[s],key=lambda r:r['episode_step'])
            provenance[s]='original_SAC_refinement_failure_prefix'
        else:
            available=sorted(full[s],key=lambda r:r['episode_step'])
            assert available,(s,'no archived controls')
            requested=origins[s];origins[s]=min(requested,available[-1]['episode_step'])
            records[s]=[r for r in available if r['episode_step']<origins[s]]
            provenance[s]=('prespecified_V311_exact_prefix' if s in (559,1714) else
                f'V311_scenario_regression_only; original_D_prefix_is_compact_{len(old_available[s])}_rows; not_the_same_failure_state; requested_step={requested}')
        assert len(records[s])==origins[s]-1,(s,len(records[s]),origins[s])
    if a.fresh:
        origins[a.scenario]=1;records[a.scenario]=[];provenance[a.scenario]='fresh_initial_state_NOT_original_failure_state'
    raw=OmegaConf.load(simplified['source_environment_config'])
    raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
    raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_video=raw.save_agent=False;raw.residual_rl.buffer_size=8
    _,cfg=suite._make_cfg(raw,checkpoint.parent.parent.name,checkpoint,list(origins));cfg.residual_rl['buffer_size']=8
    torch.set_float32_matmul_precision('high');set_seed(cfg.seed);env=make_env(cfg);cases=[]
    try:
        with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);loaded=agent.load(checkpoint)
        agent.eval();suite._apply_evaluation_stack(agent.planner,'full_mpr_mpc',loaded)
        agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
        for s,before in origins.items():
            set_seed(1+s);agent.planner.reset(seed=1+s);obs=env.reset(seed=s)
            assert int(env.unwrapped_metadrive.current_seed)==s
            info={};error=0.
            for old in records[s]:
                error=max(error,abs(np.linalg.norm(env.unwrapped_metadrive.agent.velocity)-old['mpr/actual_speed_mps']))
                obs,_,done,info=env.step(np.array(old['executed_action'],np.float32))
                if done:raise RuntimeError(f'Prefix terminated early: {s}')
            if error>1e-4:raise RuntimeError(f'Prefix mismatch: {s}, {error}')
            route0=float(info.get('route_completion',0.));rows=[]
            for i in range(a.steps):
                action=agent.act(obs,t0=False,eval_mode=True);result=agent.last_plan
                obs,reward,done,info=env.step(action)
                executed=bool(info.get('action_executed',True));certificate=result.debug.get('safety_certificate',{})
                if executed:
                    assert certificate.get('valid') and not result.safety_failure
                    assert certificate['sequence_sha256']==hashlib.sha256(result.debug['selected_mppi_actions'].contiguous().numpy().tobytes()).hexdigest()
                rows.append(dict(step=i+1,executed=executed,action=action.tolist() if executed else None,
                    speed=float(np.linalg.norm(env.unwrapped_metadrive.agent.velocity)),route=float(info.get('route_completion',0.)),
                    reward=float(reward),safety_failure=bool(info.get('safety_failure',False)),
                    crash=any(bool(info.get(k,False)) for k in ('crash','crash_vehicle','crash_object','crash_sidewalk')),
                    offroad=bool(info.get('out_of_road',False)),metrics=result.metrics,
                    decision={k:result.debug[k] for k in ('decision_state_name','baseline_certification_reason','candidate_source','selection_reason','diagnostic_state','road_rejection_kind')},
                    certificate=certificate))
                if done:break
            case=dict(scenario=s,before_step=before,state_provenance=provenance[s],replay_max_speed_error=error,records=rows,
                physical_steps=sum(r['executed'] for r in rows),unverified_action_executions=0,
                route_gain=float(info.get('route_completion',0.))-route0)
            cases.append(case);print(s,before,'physical_steps',case['physical_steps'],rows[-1]['decision'],flush=True)
    finally:env.close()
    acceptance=all(c['physical_steps']==a.steps and not any(r['crash'] or r['offroad'] or r['safety_failure'] for r in c['records']) for c in cases)
    from tools.benchmark_mppi import source_fingerprint
    a.output.write_text(json.dumps(dict(source_snapshot_sha256=source_fingerprint(ROOT),checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        cases=cases,fixed_state_acceptance_passed=acceptance,requested_closed_loop_steps=a.steps,
        note='Archived prefixes replay previously executed controls for state restoration. Newly proposed D controls execute only with a current certificate. Aborts do not demonstrate accident avoidance or recovery.'),indent=2))


if __name__=='__main__':main()
