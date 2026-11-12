"""Bounded MetaDrive negative-throttle experiments; disjoint fit/heldout scenes.

This is offline actuator identification, not an MPPI driving evaluation. Never
writes network weights or converts sparse actuator data into motion validation.
"""
import argparse, contextlib, hashlib, io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite
from planning.mppi_safety import MPPISafetyFilter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-speeds', type=float, nargs='+', default=[3., 7.])
    a = p.parse_args()
    if a.output.exists():
        p.error('Output must be a fresh directory.')
    a.output.mkdir(parents=True)
    manifest = json.loads((ROOT/'reports/simplified_tdmpc2/paired_run.json').read_text())
    checkpoint = Path(manifest['checkpoint'])
    raw = OmegaConf.load(manifest['source_environment_config'])
    raw.mpr_mpc.mppi = OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
    raw.restore_mppi_runtime = False
    raw.enable_wandb = raw.save_video = raw.save_agent = False
    raw.residual_rl.buffer_size = 8
    fit, held = [1525, 539], [1714, 559]
    _, cfg = suite._make_cfg(raw, checkpoint.parent.parent.name, checkpoint, fit+held)
    cfg.residual_rl['buffer_size'] = 8
    set_seed(1)
    env = make_env(cfg)
    records = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agent = suite._make_agent(cfg, env)
            loaded = agent.load(checkpoint)
        agent.eval()
        suite._apply_evaluation_stack(agent.planner, 'world_model_residual_rl', loaded)
        agent.set_global_step(max(agent.global_env_step, agent.planner.mppi_start_step))
        for scenario in fit+held:
            for target in a.target_speeds:
                for steer_offset in (0., .08):
                    set_seed(1+scenario); agent.planner.reset(seed=1+scenario)
                    obs = env.reset(seed=scenario)
                    assert int(env.unwrapped_metadrive.current_seed) == scenario
                    vehicle = env.unwrapped_metadrive.agent
                    acceleration = []; info = {}; done = False
                    previous_v = float(np.linalg.norm(vehicle.velocity))
                    for i in range(120):
                        action = agent.act(obs, t0=(i == 0), eval_mode=True)
                        obs, _, done, info = env.step(action)
                        speed = float(np.linalg.norm(vehicle.velocity))
                        acceleration.append((speed-previous_v)/agent.planner.control_dt)
                        previous_v = speed
                        if done or speed >= target:
                            break
                    width, wheelbase, steering = agent.planner._vehicle_geometry(vehicle)
                    context = agent.planner.local_mppi.candidate_safety.context(agent.planner.structured.planner, vehicle,
                        width=width, wheelbase=wheelbase, max_steering=steering)
                    command = np.array([np.clip(float(action[0])+steer_offset, -1, 1), -1.], np.float32)
                    before_speed = speed; deceleration = []; trace = []; traffic = []
                    start_objects = agent.planner.structured.planner._collect_obstacles(vehicle)
                    for tick in range(100):
                        if done:
                            break
                        obs, _, done, info = env.step(command)
                        speed = float(np.linalg.norm(vehicle.velocity))
                        dt = agent.planner.control_dt
                        # Exclude only the final velocity-clamped stopping tick.
                        if speed > .1:
                            deceleration.append((previous_v-speed)/dt)
                        trace.append(dict(speed=speed, xy=list(map(float, vehicle.position)), heading=float(vehicle.heading_theta),
                            dt=dt, action=command.tolist(), done=bool(done), out_of_road=bool(info.get('out_of_road', False)),
                            crash=any(bool(info.get(k, False)) for k in ('crash','crash_vehicle','crash_object','crash_sidewalk'))))
                        objects = {o.name: o for o in agent.planner.structured.planner._collect_obstacles(vehicle)}
                        t = (tick+1)*dt
                        for old in start_objects:
                            if old.name in objects:
                                obj = objects[old.name]
                                traffic.append(dict(time=t, cv_position_error=float(np.linalg.norm(obj.position-old.position-t*old.velocity)),
                                    velocity_error=float(np.linalg.norm(obj.velocity-old.velocity))))
                        previous_v = speed
                        if speed <= .1:
                            break
                    prediction_errors = {}
                    if trace:
                        controls = torch.tensor(command).view(1,1,2).expand(1,len(trace)+1,2).clone()
                        checker = agent.planner.local_mppi.candidate_safety
                        center = checker._center_rollout(controls, context)
                        legacy = checker.geometry.rollout(controls, context)
                        actual_xy = np.asarray([r['xy'] for r in trace])
                        actual_yaw = np.asarray([r['heading'] for r in trace])
                        actual_speed = np.asarray([r['speed'] for r in trace])
                        for name, prediction in [('center',center), ('legacy',legacy)]:
                            xy, yaw, speeds = [v[0,checker.geometry.substeps::checker.geometry.substeps].numpy() for v in prediction]
                            angle = yaw-actual_yaw
                            prediction_errors[name] = dict(position_m=np.linalg.norm(xy-actual_xy,axis=1).tolist(),
                                heading_rad=np.abs(np.arctan2(np.sin(angle),np.cos(angle))).tolist(),
                                speed_mps=np.abs(speeds-actual_speed).tolist())
                    record = dict(scenario=scenario, split='fit' if scenario in fit else 'heldout', target_speed=target,
                        initial_speed=before_speed, curvature=abs(float(np.tan(command[0]*steering)/wheelbase)),
                        initial_xy=context.position.tolist(), initial_heading=context.heading,
                        vehicle_signature=context.motion_signature, acceleration_samples=acceleration,
                        deceleration_samples=deceleration, trace=trace, traffic=traffic, prediction_errors=prediction_errors,
                        stopped=bool(trace and trace[-1]['speed'] <= .1),
                        clean=bool(trace and not any(r['out_of_road'] or r['crash'] for r in trace)))
                    records.append(record)
                    print(scenario, target, steer_offset, 'stopped', record['stopped'], 'clean', record['clean'], flush=True)
    finally:
        env.close()
    dataset = a.output/'measurements.json'
    dataset.write_text(json.dumps(records, indent=2))
    good_fit = [r for r in records if r['split']=='fit' and r['clean'] and r['stopped'] and r['deceleration_samples']]
    good_held = [r for r in records if r['split']=='heldout' and r['clean'] and r['stopped'] and r['deceleration_samples']]
    lower = min((min(r['deceleration_samples']) for r in good_fit), default=0.)
    upper = max((max(r['acceleration_samples']) for r in good_fit), default=0.)
    violations = [r for r in good_held if min(r['deceleration_samples']) < lower or max(r['acceleration_samples']) > upper]
    # Qualification preconditions are declared BEFORE observing heldout values.
    # Sparse clean braking samples alone cannot validate dynamic traffic or the
    # long-horizon pose envelope (the existing motion evidence has failed).
    motion = json.loads((ROOT/'reports/sac_refinement/b_motion/validation.json').read_text())
    qualified = (len(good_fit)>=8 and len(good_held)>=8 and lower>0 and not violations
                 and motion.get('validated', False) and motion.get('event_validation_supported', False))
    error_statistics = {}
    for split in ('fit','heldout'):
        for model in ('center','legacy'):
            for metric in ('position_m','heading_rad','speed_mps'):
                values=[v for r in records if r['split']==split for v in r.get('prediction_errors',{}).get(model,{}).get(metric,[])]
                error_statistics[f'{split}_{model}_{metric}'] = {f'p{q}':float(np.percentile(values,q)) if values else None for q in (50,95,99)}
    evidence = dict(schema='brake_envelope_v1', validated=bool(qualified), fit_scenes=fit, heldout_scenes=held,
        dataset=str(dataset.resolve()), dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        predictor_sha256=MPPISafetyFilter.motion_predictor_sha(), control_dt=.1,
        deceleration_lower_mps2=lower, acceleration_upper_mps2=upper,
        domain=dict(max_speed_mps=max((r['initial_speed'] for r in good_fit), default=0.),
            max_curvature_inv_m=max((r['curvature'] for r in good_fit), default=0.)),
        vehicle_signatures=list({json.dumps(r['vehicle_signature'], sort_keys=True):r['vehicle_signature'] for r in records}.values()),
        validated_prediction_time_s=0., traffic_validated=False,
        good_fit=len(good_fit), good_heldout=len(good_held), heldout_bound_violations=len(violations),
        motion_error_statistics=error_statistics,
        stop_speed_threshold_mps=.1,
        reasons=['Sparse actuator identification does not validate Center/Legacy long-horizon position envelopes or traffic acceleration bounds.',
                 'Measured fit minimum is empirical, not a proven global deceleration lower bound.'],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (a.output/'validation.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps({k:evidence[k] for k in ('validated','good_fit','good_heldout','deceleration_lower_mps2','heldout_bound_violations')}), flush=True)


if __name__=='__main__':
    main()
