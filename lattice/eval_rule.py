"""
Evaluate the pure rule-based Frenet planner without loading any RL checkpoint.

This runs FrenetSafeMetaDriveEnv with ``frenet_rl_action=False`` so the
controller uses the lattice candidate generation and nominal cost selector in
frenet_metadrive.py.
"""
import argparse
import csv
import json
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FSRL_ROOT = SCRIPT_DIR.parent.parent
for path in (SCRIPT_DIR, FSRL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval import add_bool_arg, episode_record, log_eval_episode, log_eval_summary, summarize_records
from train_sacl import SafeMetaDriveSACCfg, VALIDATION_ENV_ID, make_env_config, register_metadrive_envs


def parse_args():
    defaults = SafeMetaDriveSACCfg()
    parser = argparse.ArgumentParser(description="Evaluate the pure rule-based Frenet planner.")

    parser.add_argument("--eval-episodes", default=100, type=int)
    parser.add_argument("--num-scenarios", default=defaults.num_scenarios, type=int)
    parser.add_argument("--validation-start-seed", default=defaults.validation_start_seed, type=int)
    parser.add_argument("--horizon", default=defaults.horizon, type=int)
    parser.add_argument("--traffic-density", default=0.05, type=float)
    parser.add_argument("--accident-prob", default=defaults.accident_prob, type=float)
    add_bool_arg(parser, "crash_vehicle_done", default=defaults.crash_vehicle_done)

    parser.add_argument("--output-dir", default="rule_eval", type=str)
    add_bool_arg(parser, "tensorboard", default=True)
    parser.add_argument("--tensorboard-dir", default=None, type=str)
    add_bool_arg(parser, "save_step_data", default=False)
    add_bool_arg(parser, "render", default=False)
    parser.add_argument("--render-mode", default="topdown", choices=["topdown", "onscreen"])
    add_bool_arg(parser, "render_gif", default=True)
    add_bool_arg(parser, "save_episode_gifs", default=False)
    parser.add_argument("--episode-gif-dir", default="eval_gifs/rule", type=str)

    parser.add_argument("--frenet-target-speed", default=13.89, type=float)
    parser.add_argument("--frenet-lateral-sample-count", default=defaults.frenet_lateral_sample_count, type=int)
    parser.add_argument("--frenet-time-horizons", default=",".join(str(v) for v in defaults.frenet_time_horizons))
    parser.add_argument("--frenet-speed-offsets", default=",".join(str(v) for v in defaults.frenet_speed_offsets))
    parser.add_argument("--frenet-obstacle-cost-weight", default=18.0, type=float)
    parser.add_argument("--frenet-blocked-lane-cost-weight", default=14.0, type=float)
    parser.add_argument("--frenet-collision-penalty", default=100000.0, type=float)
    add_bool_arg(parser, "frenet_obstacle_check", default=True)
    return parser.parse_args()


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


def output_dir(args):
    base = Path(args.output_dir).expanduser()
    if not base.is_absolute():
        base = SCRIPT_DIR / base
    run_dir = base / time.strftime("rule_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_writer(args, run_dir):
    if not args.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[WARN] TensorBoard is not available. Rule metrics will not be logged.")
        return None
    if args.tensorboard_dir is None:
        tb_dir = run_dir / "tb"
    else:
        tb_dir = Path(args.tensorboard_dir).expanduser()
        if not tb_dir.is_absolute():
            tb_dir = SCRIPT_DIR / tb_dir
        tb_dir.mkdir(parents=True, exist_ok=True)
    print("tensorboard log dir: {}".format(tb_dir))
    return SummaryWriter(str(tb_dir))


def build_rule_config(args, run_dir):
    cfg = asdict(SafeMetaDriveSACCfg())
    cfg.update(
        {
            "num_scenarios": int(args.num_scenarios),
            "validation_start_seed": int(args.validation_start_seed),
            "horizon": int(args.horizon),
            "traffic_density": float(args.traffic_density),
            "accident_prob": float(args.accident_prob),
            "crash_vehicle_done": bool(args.crash_vehicle_done),
            "use_render": bool(args.render),
            "save_episode_gifs": bool(args.save_episode_gifs or (args.render and args.render_gif)),
            "episode_gif_dir": str((run_dir / args.episode_gif_dir).resolve()),
            "frenet_lateral_sample_count": int(args.frenet_lateral_sample_count),
            "frenet_time_horizons": tuple(parse_float_list(args.frenet_time_horizons)),
            "frenet_speed_offsets": tuple(parse_float_list(args.frenet_speed_offsets)),
        }
    )
    ns = types.SimpleNamespace(**cfg)
    env_config = make_env_config(ns, validation=True)
    env_config.update(
        {
            "frenet_rl_action": False,
            "frenet_target_speed": float(args.frenet_target_speed),
            "frenet_obstacle_check": bool(args.frenet_obstacle_check),
            "frenet_obstacle_cost_weight": float(args.frenet_obstacle_cost_weight),
            "frenet_blocked_lane_cost_weight": float(args.frenet_blocked_lane_cost_weight),
            "frenet_collision_penalty": float(args.frenet_collision_penalty),
            "frenet_debug_info": False,
        }
    )
    if args.render:
        env_config["episode_gif_capture_topdown"] = args.render_mode == "topdown"
        env_config["episode_gif_capture_main"] = args.render_mode == "onscreen"
    return env_config


def make_rule_env(env_config):
    return gym.make(VALIDATION_ENV_ID, config=env_config)


def safe_value(value):
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def step_record(episode_idx, step_idx, reward, terminated, truncated, info):
    keys = [
        "cost",
        "arrive_dest",
        "out_of_road",
        "crash_vehicle",
        "crash_object",
        "route_completion",
        "frenet_candidate_count",
        "frenet_valid_candidate_count",
        "frenet_obstacle_count",
        "frenet_selected_path_index",
        "frenet_selected_target_d",
        "frenet_selected_horizon",
        "frenet_selected_speed",
        "frenet_selected_collision",
        "frenet_selected_clearance",
        "frenet_reference_length",
        "frenet_lateral_bounds",
        "frenet_lane_centers",
        "frenet_vehicle_width",
    ]
    row = {
        "episode": int(episode_idx),
        "step": int(step_idx),
        "reward": float(reward),
        "terminated": float(bool(terminated)),
        "truncated": float(bool(truncated)),
    }
    for key in keys:
        row[key] = safe_value(info.get(key, 0.0))
    return row


def write_csv(path, records):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def log_rule_config(writer, env_config):
    if writer is None:
        return
    writer.add_text("rule_eval/config", json.dumps(env_config, indent=2, sort_keys=True))
    writer.flush()


def _scalar_info(info, key, default=0.0):
    value = info.get(key, default)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def log_rule_step(
    writer,
    global_step,
    episode_idx,
    episode_step,
    reward,
    episode_reward,
    episode_cost,
    terminated,
    truncated,
    info,
):
    if writer is None:
        return

    writer.add_scalar("rule_step/reward", float(reward), global_step)
    writer.add_scalar("rule_step/cost", _scalar_info(info, "cost"), global_step)
    writer.add_scalar("rule_step/episode_reward", float(episode_reward), global_step)
    writer.add_scalar("rule_step/episode_cost", float(episode_cost), global_step)
    writer.add_scalar("rule_step/episode", int(episode_idx), global_step)
    writer.add_scalar("rule_step/episode_step", int(episode_step), global_step)
    writer.add_scalar("rule_step/route_completion", _scalar_info(info, "route_completion"), global_step)
    writer.add_scalar("rule_step/success", _scalar_info(info, "arrive_dest"), global_step)
    writer.add_scalar("rule_step/out_of_road", _scalar_info(info, "out_of_road"), global_step)
    writer.add_scalar("rule_step/crash_vehicle", _scalar_info(info, "crash_vehicle"), global_step)
    writer.add_scalar("rule_step/crash_object", _scalar_info(info, "crash_object"), global_step)
    writer.add_scalar("rule_step/terminated", float(bool(terminated)), global_step)
    writer.add_scalar("rule_step/truncated", float(bool(truncated)), global_step)

    writer.add_scalar("rule_frenet/candidate_count", _scalar_info(info, "frenet_candidate_count"), global_step)
    writer.add_scalar("rule_frenet/valid_candidate_count", _scalar_info(info, "frenet_valid_candidate_count"), global_step)
    writer.add_scalar("rule_frenet/obstacle_count", _scalar_info(info, "frenet_obstacle_count"), global_step)
    writer.add_scalar("rule_frenet/selected_target_d", _scalar_info(info, "frenet_selected_target_d"), global_step)
    writer.add_scalar("rule_frenet/selected_speed", _scalar_info(info, "frenet_selected_speed"), global_step)
    writer.add_scalar("rule_frenet/selected_horizon", _scalar_info(info, "frenet_selected_horizon"), global_step)
    writer.add_scalar("rule_frenet/selected_collision", _scalar_info(info, "frenet_selected_collision"), global_step)
    writer.add_scalar("rule_frenet/selected_clearance", _scalar_info(info, "frenet_selected_clearance"), global_step)
    writer.add_scalar("rule_frenet/reference_length", _scalar_info(info, "frenet_reference_length"), global_step)
    writer.add_scalar("rule_frenet/vehicle_width", _scalar_info(info, "frenet_vehicle_width"), global_step)


def evaluate_rule(args):
    register_metadrive_envs()
    run_dir = output_dir(args)
    writer = create_writer(args, run_dir)
    env_config = build_rule_config(args, run_dir)
    log_rule_config(writer, env_config)
    write_json(run_dir / "config.json", env_config)

    env = make_rule_env(env_config)
    render_env = getattr(env, "unwrapped", env)
    records = []
    step_records = []
    global_step = 0

    print("rule eval dir: {}".format(run_dir))
    print(
        "rule eval env: scenarios={}, validation_start_seed={}, traffic_density={}, accident_prob={}".format(
            env_config["num_scenarios"],
            env_config["start_seed"],
            env_config["traffic_density"],
            env_config["accident_prob"],
        )
    )
    print("frenet_rl_action: {}".format(env_config["frenet_rl_action"]))

    try:
        for episode_idx in range(int(args.eval_episodes)):
            obs, _ = env.reset()
            del obs
            total_reward = 0.0
            total_cost = 0.0
            length = 0
            terminated = truncated = False
            last_info = {}

            while not (terminated or truncated):
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                del obs
                total_reward += float(reward)
                total_cost += float(info.get("cost", 0.0))
                length += 1
                global_step += 1
                last_info = info

                log_rule_step(
                    writer,
                    global_step,
                    episode_idx + 1,
                    length,
                    reward,
                    total_reward,
                    total_cost,
                    terminated,
                    truncated,
                    info,
                )

                if args.save_step_data:
                    step_records.append(step_record(episode_idx + 1, length, reward, terminated, truncated, info))

                if args.render:
                    if args.render_mode == "topdown":
                        render_env.render(mode="topdown", target_agent_heading_up=True)
                    else:
                        render_env.render()

            record = episode_record(total_reward, total_cost, length, terminated, truncated, last_info)
            record["episode"] = episode_idx + 1
            records.append(record)
            log_eval_episode(writer, records, record)
            print(
                "rule episode {}/{} reward={:.3f} cost={:.3f} len={} success={} out_of_road={} crash_vehicle={}".format(
                    episode_idx + 1,
                    args.eval_episodes,
                    record["reward"],
                    record["cost"],
                    int(record["length"]),
                    int(record["success"]),
                    int(record["out_of_road"]),
                    int(record["crash_vehicle"]),
                ),
                flush=True,
            )
    finally:
        env.close()

    summary = summarize_records(records)
    log_eval_summary(writer, summary, len(records))
    if writer is not None:
        writer.close()

    write_csv(run_dir / "episodes.csv", records)
    if args.save_step_data:
        write_csv(run_dir / "steps.csv", step_records)
    write_json(run_dir / "summary.json", summary)
    print("saved rule episodes: {}".format(run_dir / "episodes.csv"))
    print("saved rule summary: {}".format(run_dir / "summary.json"))
    return summary


def main():
    summary = evaluate_rule(parse_args())
    print("==================================================")
    print("Rule Frenet evaluation result")
    print("==================================================")
    for key in sorted(summary):
        print("{:<24s}: {:.6g}".format(key, summary[key]))
    print("==================================================")


if __name__ == "__main__":
    main()
