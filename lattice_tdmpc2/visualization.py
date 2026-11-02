"""Main-camera and top-down overlays for Lattice trajectory refinement."""

from __future__ import annotations

import numpy as np

from lattice.frenet_metadrive import _world_to_topdown_screen
from metadrive.utils.utils import import_pygame


pygame, _ = import_pygame()


COARSE_COLOR_RGB = (245, 190, 40)
REFINED_COLOR_RGB = (230, 45, 35)


def _draw_world_polyline(surface, renderer, points, color, width) -> None:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        return
    pixels = [_world_to_topdown_screen(renderer, point) for point in points]
    for start, end in zip(pixels[:-1], pixels[1:]):
        pygame.draw.line(surface, color, start, end, width)


def draw_refinement_overlay(raw_env, result, *, draw_refined: bool = True) -> None:
    """Draw coarse in yellow and refined in red on an existing top-down renderer."""
    renderer = getattr(raw_env, "top_down_renderer", None)
    if renderer is None or result is None:
        return
    surface = renderer.screen_canvas
    # Keep the wider coarse line visible when fallback makes both paths identical.
    _draw_world_polyline(
        surface, renderer, result.coarse_path.xy, color=COARSE_COLOR_RGB, width=6
    )
    if draw_refined:
        _draw_world_polyline(
            surface, renderer, result.refined_path.xy, color=REFINED_COLOR_RGB, width=3
        )
    renderer.blit()


class MainCameraTrajectoryOverlay:
    """Maintain transient Panda3D polylines for the current coarse/refined paths."""

    def __init__(self) -> None:
        self._nodes = []

    def update(self, raw_env, result, *, draw_refined: bool = True) -> None:
        self.clear()
        if result is None:
            return
        engine = getattr(raw_env, "engine", None)
        render = getattr(engine, "render", None)
        if render is None:
            return
        self._nodes = [
            self._draw_path(
                render, result.coarse_path.xy, COARSE_COLOR_RGB, thickness=6.0, height=0.18
            )
        ]
        if draw_refined:
            self._nodes.append(
                self._draw_path(
                    render, result.refined_path.xy, REFINED_COLOR_RGB,
                    thickness=3.0, height=0.24,
                )
            )
        self._nodes = [node for node in self._nodes if node is not None]

    def clear(self) -> None:
        for node in self._nodes:
            if not node.is_empty():
                node.remove_node()
        self._nodes = []

    @staticmethod
    def _draw_path(parent, points, color, *, thickness: float, height: float):
        from panda3d.core import LineSegs, NodePath

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            return None
        finite = np.isfinite(points[:, :2]).all(axis=1)
        points = points[finite]
        if len(points) < 2:
            return None

        line = LineSegs("lattice_tdmpc2_trajectory")
        line.set_thickness(thickness)
        line.set_color(*(channel / 255.0 for channel in color), 1.0)
        line.move_to(float(points[0, 0]), float(points[0, 1]), height)
        for point in points[1:]:
            line.draw_to(float(point[0]), float(point[1]), height)
        node = NodePath(line.create(False))
        node.reparent_to(parent)
        return node


__all__ = [
    "COARSE_COLOR_RGB",
    "REFINED_COLOR_RGB",
    "MainCameraTrajectoryOverlay",
    "draw_refinement_overlay",
]
