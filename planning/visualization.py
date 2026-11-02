"""MPR-MPC trajectory visualization kept outside the planning core."""

from __future__ import annotations

from lattice_tdmpc2.visualization import (
    MainCameraTrajectoryOverlay,
    _draw_world_polyline,
    draw_refinement_overlay as _draw_refinement_overlay,
)


FINAL_MPPI_COLOR_RGB = (30, 210, 235)


def draw_refinement_overlay(raw_env, result, *, draw_refined: bool = True) -> None:
    """Draw Lattice yellow, residual red, and selected MPPI rollout cyan."""
    _draw_refinement_overlay(raw_env, result, draw_refined=draw_refined)
    renderer = getattr(raw_env, "top_down_renderer", None)
    debug = getattr(result, "debug", {}) if result is not None else {}
    final_xy = debug.get("final_mppi_xy") if isinstance(debug, dict) else None
    if renderer is None or final_xy is None:
        return
    _draw_world_polyline(
        renderer.screen_canvas,
        renderer,
        final_xy,
        color=FINAL_MPPI_COLOR_RGB,
        width=3,
    )
    renderer.blit()


__all__ = [
    "FINAL_MPPI_COLOR_RGB",
    "MainCameraTrajectoryOverlay",
    "draw_refinement_overlay",
]
