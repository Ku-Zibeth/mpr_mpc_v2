import csv
import json
from types import SimpleNamespace

from PIL import Image
import pytest

import evaluate_paired_gif as paired


@pytest.mark.parametrize("use_mppi,has_xy", [(False, True), (True, True), (True, False), (None, True)])
def test_cyan_overlay_requires_actual_mppi_stage(monkeypatch, use_mppi, has_xy):
    from mpr_mpc.planning import visualization

    base_calls, cyan_calls, blits = [], [], []
    monkeypatch.setattr(visualization, "_draw_refinement_overlay",
                        lambda *args, **kwargs: base_calls.append(kwargs))
    monkeypatch.setattr(visualization, "_draw_world_polyline",
                        lambda *args, **kwargs: cyan_calls.append(kwargs))
    renderer = SimpleNamespace(screen_canvas=object(), blit=lambda: blits.append(True))
    raw = SimpleNamespace(top_down_renderer=renderer)
    result = SimpleNamespace(stage=SimpleNamespace(use_mppi=use_mppi),
                             debug={"final_mppi_xy": [(0., 0.), (1., 1.)] if has_xy else None})
    visualization.draw_refinement_overlay(raw, result, draw_refined=True)
    assert base_calls == [{"draw_refined": True}]
    assert len(cyan_calls) == len(blits) == int(bool(use_mppi) and has_xy)
    if cyan_calls:
        assert cyan_calls[0]["color"] == visualization.FINAL_MPPI_COLOR_RGB


def test_scenario_prefix_is_fixed_unique_and_validated(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": [1714, 1525, 539]}))
    assert paired.load_scenarios(path, 2) == [1714, 1525]
    with pytest.raises(ValueError, match="at least"):
        paired.load_scenarios(path, 4)
    for values in ([1714, 1714], [1714, True], [1714, -1]):
        path.write_text(json.dumps(values))
        with pytest.raises(ValueError):
            paired.load_scenarios(path, 2)


def test_comparison_gif_aligns_steps_and_holds_earlier_terminal_frame(tmp_path):
    recordings = []
    for side, steps in enumerate(([1, 4, 7], [1, 4, 5])):
        frames = []
        for step in steps:
            path = tmp_path / f"{side}_{step}.png"
            Image.new("RGB", (80, 80), (step * 20, side * 100, 50)).save(path)
            frames.append((step, path))
        recordings.append(frames)
    rows = [dict(final_success=0, final_route_completion=.6, end_reason="out_of_road"),
            dict(final_success=1, final_route_completion=.99, end_reason="arrive_dest")]
    output = tmp_path / "paired.gif"
    paired.write_comparison_gif(output, recordings, rows, scenario=1714, size=80, dt=.1)
    with Image.open(output) as gif:
        assert gif.size == (164, 160)
        assert gif.n_frames == 4
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info["duration"])
        assert durations == [300, 100, 200, 1500]
        final = gif.convert("RGB")
        assert final.getpixel((20, 100)) == (140, 0, 50)
        assert final.getpixel((104, 100)) == (100, 100, 50)


def test_frame_recorder_samples_steps_keeps_terminal_and_captures_overlay(tmp_path, monkeypatch):
    import pygame
    from mpr_mpc.planning import visualization

    surface = pygame.Surface((80, 80))
    calls = []

    def render(**kwargs):
        calls.append(kwargs)
        surface.fill((0, 0, 255))

    def overlay(raw, result, **kwargs):
        surface.set_at((10, 20), (255, 0, 0))

    raw = SimpleNamespace(render=render, top_down_renderer=SimpleNamespace(screen_canvas=surface))
    env = SimpleNamespace(unwrapped_metadrive=raw)
    result = SimpleNamespace(stage=SimpleNamespace(use_residual=False))
    monkeypatch.setattr(visualization, "draw_refinement_overlay", overlay)
    recorder = paired.FrameRecorder(tmp_path / "frames", 80, 3)
    recorder(1714, 1, env, result, False)
    recorder(1714, 2, env, result, False)
    recorder(1714, 3, env, result, True)
    assert [step for step, _ in recorder.frames] == [1, 3]
    assert len(calls) == 2
    assert all(call["window"] is False for call in calls)
    with Image.open(recorder.frames[-1][1]) as image:
        assert image.getpixel((10, 20)) == (255, 0, 0)


@pytest.mark.parametrize("left_stack,failures_only,outcomes,expected_gifs", [
    ("lattice_mppi", False, [(1, 1), (1, 1)], 2),
    ("lattice_sac", True, [(0, 1), (1, 1), (1, 0), (0, 0)], 3),
    ("lattice_sac", True, [(1, 1), (1, 1)], 0),
])
def test_paired_entry_writes_two_methods_per_scene_and_gifs(
    tmp_path, monkeypatch, left_stack, failures_only, outcomes, expected_gifs
):
    import pygame
    import torch
    import evaluate_model_suite as suite
    from tests.test_evaluation_ablation import diagnostic_row

    checkpoint = tmp_path / "run/models/milestone_500k_full_mpr_mpc.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    scenarios = tmp_path / "scenarios.json"
    scene_ids = [1714, 1525, 539, 559][:len(outcomes)]
    scenarios.write_text(json.dumps(scene_ids + [694]))
    args = SimpleNamespace(checkpoint=checkpoint, output=tmp_path / "output",
                           scenarios_file=scenarios, episodes=len(scene_ids), seed=1,
                           frame_stride=3, size=80, cost_coef=1.0,
                           left_stack=left_stack, failures_only=failures_only)
    calls = []
    surface = pygame.Surface((80, 80))

    def render(**kwargs):
        surface.fill((30, 150, 70))

    env = SimpleNamespace(unwrapped_metadrive=SimpleNamespace(
        render=render, top_down_renderer=SimpleNamespace(screen_canvas=surface)))

    def evaluate(raw, version, path, ids, *, on_episode, on_step):
        calls.append((raw.evaluation_stack, ids[0], raw.seed))
        assert raw.benchmark.paired_episode_seeds
        assert raw.benchmark.mppi_eval_seed == 1
        assert raw.benchmark.mppi_cost_coef == 1.0
        side = int(raw.evaluation_stack == "full_mpr_mpc")
        success = outcomes[scene_ids.index(ids[0])][side]
        row = diagnostic_row(ids[0], raw.seed, float(success))
        row.update(end_reason="arrive_dest" if success else "out_of_road",
                   episode_seed=ids[0] + 1)
        if raw.evaluation_stack == "full_mpr_mpc":
            row["mean_mpr_sac_inference_cpu_wall_ms"] = 1.25
        on_step(ids[0], 1, env, None, False)
        on_step(ids[0], 2, env, None, True)
        on_episode(row)
        return [row], {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(suite, "_make_cfg", lambda raw, *_: (raw, raw))
    monkeypatch.setattr(suite, "_evaluate_checkpoint", evaluate)
    paired.run(args)
    expected_left = "lattice_mppi" if left_stack == "lattice_mppi" else "world_model_residual_rl"
    assert calls == [(stack, scene, 1) for scene in scene_ids
                     for stack in [expected_left, "full_mpr_mpc"]]
    gifs = list((args.output / "gifs").glob("*.gif"))
    assert len(gifs) == expected_gifs
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["left"] == ("Lattice + MPPI" if left_stack == "lattice_mppi" else "Lattice + SAC")
    assert manifest["right"] == "Lattice + SAC + MPPI"
    assert manifest["failures_only"] == failures_only
    summaries = json.loads((args.output / "summary.json").read_text())
    assert all(row["episodes"] == row["unique_scenarios"] == len(scene_ids) for row in summaries)
    with (args.output / "episodes.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2 * len(scene_ids)
    with (args.output / "gif_index.csv").open() as stream:
        index_rows = list(csv.DictReader(stream))
    assert len(index_rows) == len(scene_ids)
    for row, (left, right) in zip(index_rows, outcomes):
        keep = not failures_only or not (left and right)
        assert row["gif_saved"] == str(keep)
        assert bool(row["gif"]) == keep
        if keep:
            assert (args.output / row["gif"]).is_file()
    assert (args.output / "paired_comparisons.csv").is_file()
    assert not list(args.output.glob("frames_*"))
    with pytest.raises(FileExistsError):
        paired.run(args)
