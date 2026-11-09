"""Paired planner evaluations with simulation-time-aligned side-by-side GIFs."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
VARIANTS = (
    ("C_lattice_mppi", "lattice_mppi", "Lattice + MPPI"),
    ("D_lattice_sac_mppi", "full_mpr_mpc", "Lattice + SAC + MPPI"),
)


def comparison_variants(left_stack):
    if left_stack == "lattice_mppi":
        return VARIANTS
    if left_stack == "lattice_sac":
        return (("B_lattice_sac", "world_model_residual_rl", "Lattice + SAC"), VARIANTS[1])
    raise ValueError(f"Unsupported left stack: {left_stack}")


def load_scenarios(path: Path, episodes: int) -> list[int]:
    payload = json.loads(path.read_text())
    values = payload.get("scenarios", []) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or episodes <= 0 or len(values) < episodes:
        raise ValueError("Scenario file must contain at least --episodes IDs.")
    scenarios = values[:episodes]
    if any(type(seed) is not int or seed < 0 for seed in scenarios):
        raise ValueError("Scenario IDs must be nonnegative integers.")
    if len(set(scenarios)) != episodes:
        raise ValueError("Selected scenario IDs must be unique.")
    return scenarios


class FrameRecorder:
    def __init__(self, directory: Path, size: int, stride: int):
        self.directory, self.size, self.stride = directory, size, stride
        self.directory.mkdir(parents=True)
        self.frames = []

    def __call__(self, scenario, step, env, result, done):
        if (step - 1) % self.stride and not done:
            return
        import numpy as np
        import pygame
        from mpr_mpc.planning.visualization import draw_refinement_overlay

        raw_env = env.unwrapped_metadrive
        raw_env.render(
            mode="topdown", window=False, screen_record=False,
            screen_size=(self.size, self.size), film_size=(2000, 2000),
            scaling=5, target_agent_heading_up=True,
            draw_target_vehicle_trajectory=True,
        )
        if result is not None:
            draw_refinement_overlay(raw_env, result, draw_refined=result.stage.use_residual)
        # Read RGB after the overlay; render() returned an earlier surface copy.
        pixels = pygame.surfarray.array3d(raw_env.top_down_renderer.screen_canvas)
        image = Image.fromarray(np.transpose(pixels, (1, 0, 2)))
        path = self.directory / f"{step:06d}.png"
        image.save(path)
        image.close()
        self.frames.append((step, path))


def write_comparison_gif(path, recordings, rows, *, scenario, size, dt, variants=VARIANTS):
    if len(recordings) != 2 or any(not frames for frames in recordings):
        raise ValueError("Both methods must have recorded frames.")
    steps = [[step for step, _ in frames] for frames in recordings]
    timeline = sorted(set(steps[0]) | set(steps[1]))
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    images, durations = [], []
    try:
        for index, step in enumerate(timeline):
            image = Image.new("RGB", (size * 2 + 4, size + 80), "#ffffff")
            draw = ImageDraw.Draw(image)
            for side, (_, _, title) in enumerate(variants):
                frame_index = max(0, bisect_right(steps[side], step) - 1)
                source = recordings[side][frame_index][1]
                x = side * (size + 4)
                with Image.open(source) as frame:
                    if frame.size != (size, size):
                        raise ValueError("Unexpected frame dimensions.")
                    image.paste(frame.convert("RGB"), (x, 80))
                ended = step >= steps[side][-1]
                row = rows[side]
                status = (f"END: {row.get('end_reason') or 'done'}" if ended
                          else f"step={step}  t={step * dt:.1f}s")
                draw.text((x + 8, 5), title, fill="#111111", font=font)
                draw.text((x + 8, 28), f"scenario={scenario}  {status}", fill="#222222", font=font)
                if ended:
                    outcome = (f"success={int(row['final_success'])}  "
                               f"route={row['final_route_completion']:.1%}")
                    draw.text((x + 8, 51), outcome, fill="#222222", font=font)
            images.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
            image.close()
            duration = (timeline[index + 1] - step) * dt if index + 1 < len(timeline) else 1.5
            durations.append(max(10, round(duration * 1000 / 10) * 10))
        images[0].save(
            path, save_all=True, append_images=images[1:], duration=durations,
            loop=0, optimize=False, disposal=2,
        )
    finally:
        for image in images:
            image.close()


def run(args):
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    import evaluate_model_suite as suite
    from evaluate_ablation import _EpisodeCSVWriter
    from evaluation_statistics import paired_comparisons, summarize_runs

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA.")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    scenarios = load_scenarios(args.scenarios_file, args.episodes)
    variants = comparison_variants(args.left_stack)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "episodes.csv").exists():
        raise FileExistsError("Choose a fresh output directory; existing results will not be overwritten.")
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT)):
        raw = compose(config_name="config", overrides=["+experiment=E5"])
    with open_dict(raw):
        raw.checkpoint = str(checkpoint)
        raw.restore_checkpoint_runtime = True
        raw.seed = args.seed
        raw.eval_episodes = len(scenarios)
        raw.enable_wandb = raw.save_video = False
        raw.metadrive.simulator.start_seed = 300
        raw.metadrive.simulator.num_scenarios = max(2000, max(scenarios) - 300 + 1)
        raw.benchmark = dict(paired_episode_seeds=True, mppi_eval_seed=args.seed,
                             mppi_cost_coef=args.cost_coef)
    effective, _ = suite._make_cfg(raw, checkpoint.parent.parent.name, checkpoint, scenarios)
    simulator = effective.metadrive.simulator
    dt = float(simulator.physics_world_step_size) * int(simulator.decision_repeat)
    OmegaConf.save(effective, output / "config.yaml")
    (output / "scenarios.json").write_text(json.dumps({"scenarios": scenarios}, indent=2))
    manifest = dict(
        checkpoint=str(checkpoint), scenarios=scenarios, seed=args.seed,
        episode_seed_rule="(test_seed + scenario) modulo 2**32",
        left=variants[0][2], right=variants[1][2], control_dt=dt,
        frame_stride=args.frame_stride, panel_size=args.size, cost_coef=args.cost_coef,
        failures_only=args.failures_only, failure_definition="final_success != 1",
        alignment="simulation steps; hold final frame of the method that finishes earlier",
        episodes_per_method=len(scenarios), total_episodes=2 * len(scenarios),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    gif_dir = output / "gifs"
    gif_dir.mkdir()
    print(f"Checkpoint: {checkpoint}\nScenarios: {scenarios}\nOutput: {output}", flush=True)
    all_rows, gif_index = [], []
    saved_gifs = 0
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = _EpisodeCSVWriter(stream)
        for index, scenario in enumerate(scenarios, 1):
            with tempfile.TemporaryDirectory(prefix=f"frames_{scenario}_", dir=output) as temporary:
                recordings, scene_rows = [], []
                for label, stack, title in variants:
                    print(f"\n[{index:02d}/{len(scenarios):02d}] scenario={scenario}, {title}", flush=True)
                    variant = OmegaConf.create(OmegaConf.to_container(raw, resolve=True))
                    variant.evaluation_stack = stack
                    recorder = FrameRecorder(Path(temporary) / label, args.size, args.frame_stride)

                    def save_episode(row):
                        row.update(configuration=label, test_seed=args.seed)
                        writer.writerow(row)
                        all_rows.append(row)

                    rows, _ = suite._evaluate_checkpoint(
                        variant, checkpoint.parent.parent.name, checkpoint, [scenario],
                        on_episode=save_episode, on_step=recorder,
                    )
                    recordings.append(recorder.frames)
                    scene_rows.append(rows[0])
                failed = any(float(row["final_success"]) != 1.0 for row in scene_rows)
                keep = not args.failures_only or failed
                gif = gif_dir / f"{index:02d}_scenario_{scenario}.gif"
                if keep:
                    write_comparison_gif(gif, recordings, scene_rows,
                                         scenario=scenario, size=args.size, dt=dt, variants=variants)
                    saved_gifs += 1
                    print(f"Saved GIF: {gif}", flush=True)
                else:
                    print(f"Both methods succeeded; no GIF saved for scenario={scenario}.", flush=True)
                gif_index.append(dict(
                    scenario=scenario, left=variants[0][2], right=variants[1][2],
                    left_success=scene_rows[0]["final_success"],
                    right_success=scene_rows[1]["final_success"],
                    left_end_reason=scene_rows[0].get("end_reason", ""),
                    right_end_reason=scene_rows[1].get("end_reason", ""),
                    any_failure=failed, gif_saved=keep,
                    gif=str(gif.relative_to(output)) if keep else "",
                ))
            summaries = [summarize_runs(label, [r for r in all_rows if r['configuration'] == label],
                                       samples=2000, seed=args.seed) for label, _, _ in variants]
            suite._write_csv(output / "summary.csv", summaries)
            suite._write_csv(output / "gif_index.csv", gif_index)
            suite._write_csv(output / "paired_comparisons.csv", paired_comparisons(
                all_rows, samples=2000, seed=args.seed, pairs=[(variants[0][0], variants[1][0])],
            ))
            (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    print(f"\nFinished: {len(scenarios)} paired scenarios, {saved_gifs} comparison GIFs\nResults: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenarios-file", type=Path, default=ROOT / "scenarios/fixed100.json")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--cost-coef", type=float, default=1.0)
    parser.add_argument("--left-stack", choices=("lattice_mppi", "lattice_sac"), default="lattice_mppi",
                        help="Left method; right method is always Lattice + SAC + MPPI.")
    parser.add_argument("--failures-only", action="store_true",
                        help="Save a GIF only if at least one method has final_success != 1.")
    args = parser.parse_args()
    if args.size < 400 or args.frame_stride < 1 or not 0 <= args.seed < 2**32:
        parser.error("Require size >= 400, frame-stride >= 1, and a nonnegative 32-bit seed.")
    if not 0 <= args.cost_coef < float("inf"):
        parser.error("cost-coef must be finite and nonnegative.")
    run(args)


if __name__ == "__main__":
    main()
