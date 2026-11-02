"""Record matched-scenario Lattice versus residual-policy comparison GIFs."""

from __future__ import annotations

import copy
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
UPSTREAM_ROOT = REPO_ROOT / "tdmpc2"
for path in (REPO_ROOT, UPSTREAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from env import make_env
from lattice.frenet_metadrive import FRENET_DEFAULT_CONFIG, MetaDriveFrenetController
from lattice_tdmpc2.evaluate_visual import _resolve_checkpoints, _visual_config
from lattice_tdmpc2.planner import ResidualSACLatticePlanner
from lattice_tdmpc2.trainer import load_tdmpc2, make_sac, seed_everything
from lattice_tdmpc2.versions import training_semantics
from lattice_tdmpc2.visualization import MainCameraTrajectoryOverlay, draw_refinement_overlay
from metadrive.engine.core.main_camera import MainCamera
from metadrive.obs.top_down_obs_impl import WorldSurface


@dataclass
class RolloutSummary:
    scenario: int
    policy: str
    episode_return: float
    episode_cost: float
    success: float
    route_completion: float
    length: int
    end_reason: str


class VideoSink:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = float(fps)
        self.writer = None
        self.size = None

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        height, width = frame.shape[:2]
        size = (width - width % 2, height - height % 2)
        if size != (width, height):
            frame = frame[:size[1], :size[0]]
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.path}")
            self.size = size
        elif size != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def _sample_scenarios(cfg, count: int, seed: int, limit: int) -> list[int]:
    simulator = cfg.metadrive["simulator"]
    training_start = int(simulator["start_seed"])
    training_end = training_start + int(simulator["num_scenarios"])
    candidates = np.arange(limit, dtype=np.int64)
    candidates = candidates[(candidates < training_start) | (candidates >= training_end)]
    if count <= 0 or count > len(candidates):
        raise ValueError(f"comparison.episodes must be in [1, {len(candidates)}].")
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.choice(candidates, size=count, replace=False)]


def _controller(cfg):
    lattice_config = dict(FRENET_DEFAULT_CONFIG)
    lattice_config.update(dict(cfg.lattice))
    lattice_config["_tracking_target_speed"] = training_semantics(cfg).tracking_target_speed
    return MetaDriveFrenetController(lattice_config)


def _hud(policy: str, scenario: int, step: int, result, episode_return: float, episode_cost: float):
    return {
        "policy": policy,
        "scenario": scenario,
        "step": step,
        "coarse d/v": f"{result.coarse_d:+.2f} m / {result.coarse_v:.2f} m/s",
        "delta d/v": f"{result.delta_d:+.2f} m / {result.delta_v:+.2f} m/s",
        "output d/v": f"{result.refined_d:+.2f} m / {result.refined_v:.2f} m/s",
        "return": f"{episode_return:.2f}",
        "cost": f"{episode_cost:.2f}",
        "fallback": result.fallback_reason or "none",
    }


def _capture_frames(raw_env, overlay, result, text, *, ours: bool):
    overlay.update(raw_env, result, draw_refined=ours)
    raw_env.render(text=text)
    main_rgb = raw_env.main_camera.perceive(to_float=False)

    raw_env.render(
        text={},
        mode="topdown",
        window=False,
        screen_record=False,
        screen_size=(720, 720),
        film_size=(1200, 1200),
        scaling=8,
        num_stack=15,
        history_smooth=1,
        target_agent_heading_up=True,
        draw_target_vehicle_trajectory=True,
    )
    draw_refinement_overlay(raw_env, result, draw_refined=ours)
    bev_rgb = WorldSurface.to_cv2_image(raw_env.top_down_renderer.screen_canvas)
    return (
        cv2.cvtColor(np.ascontiguousarray(main_rgb), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.ascontiguousarray(bev_rgb), cv2.COLOR_RGB2BGR),
    )


def _rollout(
    env,
    planner,
    sac_agent,
    scenario: int,
    policy: str,
    main_path: Path,
    bev_path: Path,
    *,
    fps: float,
    frame_stride: int,
    max_steps: int | None,
) -> RolloutSummary:
    observation = env.reset(seed=scenario)
    planner.reset()
    context = planner.prepare(observation, env.unwrapped_metadrive.agent)
    overlay = MainCameraTrajectoryOverlay()
    main_sink = VideoSink(main_path, fps)
    bev_sink = VideoSink(bev_path, fps)
    episode_return = 0.0
    episode_cost = 0.0
    step = 0
    done = False
    info = {}
    try:
        while not done and (max_steps is None or step < max_steps):
            normalized_action = (
                sac_agent.act(context.residual_state.numpy(), deterministic=True)
                if sac_agent is not None else np.zeros(2, dtype=np.float32)
            )
            result = planner.refine(context, normalized_action, env.unwrapped_metadrive.agent)
            if step % frame_stride == 0:
                text = _hud(policy, scenario, step, result, episode_return, episode_cost)
                main_frame, bev_frame = _capture_frames(
                    env.unwrapped_metadrive, overlay, result, text, ours=sac_agent is not None
                )
                main_sink.write(main_frame)
                bev_sink.write(bev_frame)
            observation, reward, done, info = env.step(result.low_level_action)
            episode_return += float(reward)
            episode_cost += float(info.get("cost", 0.0))
            step += 1
            if not done and (max_steps is None or step < max_steps):
                context = planner.prepare(observation, env.unwrapped_metadrive.agent)
    finally:
        overlay.clear()
        main_sink.close()
        bev_sink.close()
    return RolloutSummary(
        scenario=scenario,
        policy=policy,
        episode_return=episode_return,
        episode_cost=episode_cost,
        success=float(info.get("success", 0.0)),
        route_completion=float(info.get("route_completion", 0.0)),
        length=step,
        end_reason=str(info.get("end_reason", "max_steps" if not done else "unknown")),
    )


def _resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (frame.shape[1], frame.shape[0]) == size:
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _comparison_panel(
    bev: np.ndarray,
    main: np.ndarray,
    label: str,
    *,
    panel_size: int,
    inset_width: int,
) -> np.ndarray:
    panel = _resize(bev, (panel_size, panel_size)).copy()
    cv2.rectangle(panel, (0, 0), (panel_size, 46), (20, 20, 20), -1)
    cv2.putText(
        panel, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    inset_height = max(int(round(main.shape[0] * inset_width / main.shape[1])), 1)
    inset = _resize(main, (inset_width, inset_height))
    x0, y0 = 12, 56
    x1 = min(x0 + inset.shape[1], panel_size - 12)
    y1 = min(y0 + inset.shape[0], panel_size - 12)
    inset = inset[:y1 - y0, :x1 - x0]
    cv2.rectangle(panel, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (255, 255, 255), 3)
    panel[y0:y1, x0:x1] = inset
    cv2.putText(
        panel, "MAIN CAMERA", (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return panel


def _ffmpeg_binary() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    candidates = sorted(
        (Path(sys.prefix) / "lib").glob(
            "python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"
        )
    )
    if not candidates:
        raise RuntimeError("ffmpeg is required to encode GIF output.")
    return str(candidates[-1])


def _encode_gif(video_path: Path, gif_path: Path, fps: float, colors: int) -> None:
    colors = min(max(int(colors), 2), 256)
    palette_filter = (
        f"fps={fps:.6f},split[s0][s1];"
        f"[s0]palettegen=max_colors={colors}:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
    )
    subprocess.run(
        [
            _ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path), "-filter_complex", palette_filter,
            "-loop", "0", str(gif_path),
        ],
        check=True,
    )


def _merge_to_gif(
    lattice_main_path: Path,
    lattice_bev_path: Path,
    ours_main_path: Path,
    ours_bev_path: Path,
    output_path: Path,
    *,
    fps: float,
    panel_size: int,
    inset_width: int,
    colors: int,
) -> None:
    captures = [
        cv2.VideoCapture(str(path))
        for path in (
            lattice_main_path, lattice_bev_path, ours_main_path, ours_bev_path
        )
    ]
    composite_path = output_path.with_suffix(".temporary.mp4")
    sink = VideoSink(composite_path, fps)
    last_frames = [None, None, None, None]
    try:
        while True:
            reads = [capture.read() for capture in captures]
            for index, (ok, frame) in enumerate(reads):
                if ok:
                    last_frames[index] = frame
            if not any(ok for ok, _ in reads):
                break
            if any(frame is None for frame in last_frames):
                raise RuntimeError("A comparison rollout produced an empty video.")
            lattice_main, lattice_bev, ours_main, ours_bev = last_frames
            frame = np.hstack([
                _comparison_panel(
                    lattice_bev, lattice_main, "LATTICE",
                    panel_size=panel_size, inset_width=inset_width,
                ),
                _comparison_panel(
                    ours_bev, ours_main, "OURS (V4)",
                    panel_size=panel_size, inset_width=inset_width,
                ),
            ])
            sink.write(frame)
    finally:
        for capture in captures:
            capture.release()
        sink.close()
    try:
        _encode_gif(composite_path, output_path, fps, colors)
    finally:
        composite_path.unlink(missing_ok=True)


def _write_summary(output_dir: Path, scenarios: list[int], rows: list[RolloutSummary], metadata: dict) -> None:
    with (output_dir / "scenarios.json").open("w", encoding="utf-8") as stream:
        json.dump({"scenarios": scenarios, **metadata}, stream, indent=2)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(RolloutSummary.__dataclass_fields__)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    sac_checkpoint, wm_checkpoint, sac_state, saved_cfg = _resolve_checkpoints(raw_cfg)
    cfg = _visual_config(raw_cfg, saved_cfg, sac_checkpoint, wm_checkpoint)
    comparison = dict(raw_cfg.comparison)
    episodes = int(comparison["episodes"])
    scenario_seed = int(comparison["scenario_seed"])
    scenario_limit = int(comparison["scenario_seed_limit"])
    frame_stride = max(int(comparison["frame_stride"]), 1)
    fps = float(comparison["fps"]) / frame_stride
    panel_size = int(comparison["panel_size"])
    inset_width = int(comparison["main_inset_width"])
    gif_colors = int(comparison["gif_colors"])
    if panel_size < 320:
        raise ValueError("comparison.panel_size must be at least 320.")
    if inset_width <= 0 or inset_width >= panel_size - 24:
        raise ValueError("comparison.main_inset_width must fit inside one BEV panel.")
    max_steps_value = comparison.get("max_steps")
    max_steps = None if max_steps_value is None else int(max_steps_value)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("comparison.max_steps must be positive or null.")

    seed_everything(int(cfg.seed))
    scenarios = _sample_scenarios(cfg, episodes, scenario_seed, scenario_limit)
    cfg.metadrive["simulator"]["start_seed"] = 0
    cfg.metadrive["simulator"]["num_scenarios"] = scenario_limit
    cfg.metadrive["simulator"]["use_render"] = False
    cfg.metadrive["simulator"]["image_observation"] = True
    cfg.metadrive["simulator"]["state_observation_with_offscreen_render"] = True
    cfg.metadrive["simulator"]["window_size"] = (640, 360)
    cfg.metadrive["simulator"]["sensors"] = {
        "main_camera": (MainCamera, 640, 360),
    }
    cfg.metadrive["simulator"]["show_interface"] = False
    cfg.metadrive["simulator"]["force_render_fps"] = None
    cfg.metadrive["simulator"]["show_logo"] = False
    cfg.metadrive["simulator"]["show_fps"] = False

    output_root = Path(hydra.utils.get_original_cwd()) / str(comparison["output_dir"])
    run_name = (
        f"lattice_vs_v4_gif_step_{sac_state.get('environment_step', 'unknown')}"
        f"_seed_{scenario_seed}"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Matched scenarios: {scenarios}")
    print(f"Output directory: {output_dir}")
    print("GIF layout: LEFT=LATTICE, RIGHT=OURS; BEV background with main-camera inset")
    print("Rendering mode: offscreen (no visualization windows)")

    rows: list[RolloutSummary] = []
    with tempfile.TemporaryDirectory(prefix="lattice_vs_ours_", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        lattice_files = {}
        lattice_cfg = copy.deepcopy(cfg)
        lattice_cfg.planner["type"] = "lattice"
        lattice_env = make_env(lattice_cfg)
        try:
            lattice_planner = ResidualSACLatticePlanner(
                lattice_cfg, _controller(lattice_cfg), action_space=lattice_env.action_space
            )
            for index, scenario in enumerate(scenarios, start=1):
                main_path = temporary_dir / f"{scenario}_lattice_main.mp4"
                bev_path = temporary_dir / f"{scenario}_lattice_bev.mp4"
                rows.append(_rollout(
                    lattice_env, lattice_planner, None, scenario, "Lattice",
                    main_path, bev_path, fps=fps, frame_stride=frame_stride,
                    max_steps=max_steps,
                ))
                lattice_files[scenario] = (main_path, bev_path)
                print(f"[{index:02d}/{episodes:02d}] Lattice seed={scenario} complete")
        finally:
            lattice_env.close()

        ours_cfg = copy.deepcopy(cfg)
        ours_env = make_env(ours_cfg)
        try:
            tdmpc_agent = load_tdmpc2(ours_cfg, str(ours_cfg.planner["type"]))
            ours_planner = ResidualSACLatticePlanner(
                ours_cfg, _controller(ours_cfg), tdmpc_agent=tdmpc_agent,
                action_space=ours_env.action_space,
            )
            observation = ours_env.reset(seed=scenarios[0])
            ours_planner.reset()
            context = ours_planner.prepare(observation, ours_env.unwrapped_metadrive.agent)
            saved_residual = (sac_state.get("config") or {}).get("residual_rl", {})
            sac_agent = make_sac(
                ours_cfg, int(context.residual_state.numel()),
                torch.device(str(ours_cfg.residual_rl["device"])),
                str(ours_cfg.planner["type"]),
                hidden_sizes=saved_residual.get("hidden_sizes", ours_cfg.residual_rl["hidden_sizes"]),
            )
            sac_agent.load(sac_checkpoint, load_optimizers=False)
            for index, scenario in enumerate(scenarios, start=1):
                ours_main = temporary_dir / f"{scenario}_ours_main.mp4"
                ours_bev = temporary_dir / f"{scenario}_ours_bev.mp4"
                rows.append(_rollout(
                    ours_env, ours_planner, sac_agent, scenario, "Ours (V4)",
                    ours_main, ours_bev, fps=fps, frame_stride=frame_stride,
                    max_steps=max_steps,
                ))
                lattice_main, lattice_bev = lattice_files[scenario]
                _merge_to_gif(
                    lattice_main, lattice_bev, ours_main, ours_bev,
                    output_dir / f"scenario_{scenario:04d}.gif",
                    fps=fps, panel_size=panel_size, inset_width=inset_width,
                    colors=gif_colors,
                )
                lattice_main.unlink(missing_ok=True)
                lattice_bev.unlink(missing_ok=True)
                ours_main.unlink(missing_ok=True)
                ours_bev.unlink(missing_ok=True)
                print(f"[{index:02d}/{episodes:02d}] Ours seed={scenario} saved")
        finally:
            ours_env.close()

    _write_summary(
        output_dir, scenarios, rows,
        {
            "sac_checkpoint": str(sac_checkpoint),
            "world_model_checkpoint": str(wm_checkpoint),
            "safe_fallback": bool(cfg.residual_rl["safe_fallback"]),
            "traffic_density": float(cfg.metadrive["simulator"]["traffic_density"]),
            "accident_prob": float(cfg.metadrive["simulator"]["accident_prob"]),
            "fps": fps,
            "frame_stride": frame_stride,
            "panel_size": panel_size,
            "main_inset_width": inset_width,
            "gif_colors": gif_colors,
            "rendering": "offscreen",
        },
    )
    print(f"Comparison complete: {output_dir}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
