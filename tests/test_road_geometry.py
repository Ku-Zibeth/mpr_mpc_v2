from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box
from shapely.ops import unary_union

from mpr_mpc.planning import safety


def road_context(polygons, evaluator=None):
    evaluator = evaluator or safety.TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    lanes = [SimpleNamespace(index=("a", "b", i), shapely_polygon=polygon)
             for i, polygon in enumerate(polygons)]
    navigation = SimpleNamespace(current_ref_lanes=lanes, next_ref_lanes=[])
    vehicle = SimpleNamespace(navigation=navigation, lane=lanes[0],
        position=np.array([10., 10.]), heading_theta=0., velocity=np.zeros(2),
        current_action=[0., 0.], LENGTH=4.)
    planner = SimpleNamespace(_navigation_lane_groups=lambda nav: [lanes],
        _reference_lane_sequence=lambda v: lanes, _collect_obstacles=lambda v: [])
    return evaluator, planner, vehicle


def get_context(evaluator, planner, vehicle):
    return evaluator.context(planner, vehicle, width=2., wheelbase=2.5, max_steering=.6)


def test_self_intersecting_lane_is_repaired_before_union(monkeypatch):
    bowtie = Polygon([(0, 0), (4, 4), (0, 4), (4, 0)])
    assert not bowtie.is_valid
    calls = []

    def checked_union(polygons):
        assert all(p.is_valid and p.geom_type == "Polygon" for p in polygons)
        calls.append(len(polygons))
        return unary_union(polygons)

    monkeypatch.setattr(safety, "unary_union", checked_union)
    evaluator, planner, vehicle = road_context([bowtie, box(4, 0, 8, 4)])
    ctx = get_context(evaluator, planner, vehicle)
    assert ctx.road.is_valid and ctx.road.area == pytest.approx(24)
    assert ctx.road.covers(Point(2, .5))
    assert ctx.road.covers(Point(2, 3.5))
    assert not ctx.road.covers(Point(.5, 2))
    assert get_context(evaluator, planner, vehicle).road is ctx.road
    assert len(calls) == 1


def test_collapsed_line_artifact_is_not_drivable():
    polygon = Polygon([(0, 0), (4, 0), (4, 4), (2, 4), (2, 6), (2, 4), (0, 4)])
    evaluator, planner, vehicle = road_context([polygon])
    ctx = get_context(evaluator, planner, vehicle)
    assert ctx.road.geom_type == "Polygon"
    assert ctx.road.area == pytest.approx(16)
    assert not ctx.road.covers(Point(2, 5))


def test_repair_preserves_holes_and_offroad_rejection():
    road = Polygon(box(0, 0, 30, 30).exterior.coords,
                   [box(8, 8, 12, 12).exterior.coords])
    supplied = GeometryCollection([road, LineString([(30, 15), (40, 15)])])
    evaluator, planner, vehicle = road_context([supplied])
    ctx = get_context(evaluator, planner, vehicle)
    assert ctx.road.area == pytest.approx(884)
    assert not ctx.road.covers(Point(10, 10))
    assert not ctx.road.covers(Point(35, 15))
    result = evaluator.check(torch.zeros(1, 11, 2), ctx)
    assert result.offroad.item() and not result.valid.item()


@pytest.mark.parametrize("geometry", [
    None,
    Polygon(),
    LineString([(0, 0), (1, 1)]),
    Polygon([(0, 0), (1, 0), (2, 0)]),
    Polygon([(0, 0), (1, 0), (1, np.inf), (0, 1)]),
])
def test_unusable_lane_fails_explicitly_without_caching(geometry):
    evaluator, planner, vehicle = road_context([geometry, box(0, 0, 30, 30)])
    with pytest.raises(RuntimeError, match="road|lane"):
        get_context(evaluator, planner, vehicle)
    assert not evaluator._road_cache
