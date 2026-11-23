"""Opt-in body-distance road departure rule for controlled evaluation."""
from __future__ import annotations

import numpy as np
from shapely import get_coordinates, make_valid
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep


def _areas(geometry):
    if geometry is None or geometry.is_empty or not np.isfinite(get_coordinates(geometry)).all():
        raise RuntimeError('Invalid MetaDrive road boundary geometry')
    pending = [geometry if geometry.is_valid else make_valid(geometry)]
    parts = []
    while pending:
        item = pending.pop()
        if item.geom_type == 'Polygon' and not item.is_empty:
            parts.append(item)
        elif item.geom_type in ('MultiPolygon', 'GeometryCollection'):
            pending.extend(item.geoms)
    if not parts:
        raise RuntimeError('Road boundary has no drivable area')
    return parts


class BodyMarginRoadBoundary:
    """Cover the REAL body polygon with a buffered route-road corridor.

    Use every lane of each route road, so lane dividers do not become road
    edges. Opposing/unrelated roads are excluded. Route joins are unioned
    before buffering, avoiding false departure at a lane/road seam.
    """
    def __init__(self):
        self._map = None
        self._key = None
        self._road = None
        self._buffered = None
        self._margin = None

    def check(self, vehicle, margin_widths=0.0, *, margin_m=None):
        fraction = float(margin_widths)
        width = float(vehicle.WIDTH)
        if not np.isfinite([fraction, width]).all() or fraction < 0 or width <= 0:
            raise ValueError('Road margin must be finite/nonnegative; actual vehicle width must be positive')
        if margin_m is not None:
            if not np.isfinite(margin_m) or margin_m < 0:
                raise ValueError('Absolute road margin in metres must be finite/nonnegative')
            if fraction != 0:
                raise ValueError('Choose either metres or vehicle-width margin, not both')
        margin = fraction * width if margin_m is None else float(margin_m)
        navigation = vehicle.navigation
        road_map = navigation.map
        route = tuple(navigation.checkpoints)
        if len(route) < 2:
            raise RuntimeError('Body-margin departure requires a valid navigation route')
        if self._map is not road_map or self._key != route:
            graph = road_map.road_network.graph
            lanes = [lane for start, end in zip(route[:-1], route[1:]) for lane in graph[start][end]]
            polygons = [part for lane in lanes for part in _areas(lane.shapely_polygon)]
            self._road = unary_union(polygons)
            if not self._road.is_valid:
                self._road = unary_union(_areas(self._road))
            if self._road.is_empty or not self._road.is_valid:
                raise RuntimeError('Unable to construct route-road boundary')
            self._map, self._key = road_map, route
            self._margin = None
        if self._margin != margin:
            self._buffered = prep(self._road.buffer(margin, quad_segs=32) if margin else self._road)
            self._margin = margin
        # Native bounding_box already uses world XY and the actual model's
        # dimensions/heading. Do not confuse width with length or speed units.
        footprint = Polygon(vehicle.bounding_box)
        if footprint.is_empty or not footprint.is_valid or not np.isfinite(get_coordinates(footprint)).all():
            raise RuntimeError('Invalid real vehicle body geometry')
        return not self._buffered.covers(footprint), margin
