"""Tree polygon, its convex decomposition and placement transforms.

A placement (x, y, deg) puts the tree's local origin -- the centre of the top of the
trunk -- at (x, y) and rotates the tree counter-clockwise by ``deg`` degrees about it.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
from shapely.geometry import Polygon

TRUNK_W = Decimal("0.15")
TRUNK_H = Decimal("0.2")
BASE_W = Decimal("0.7")
MID_W = Decimal("0.4")
TOP_W = Decimal("0.25")
TIP_Y = Decimal("0.8")
TIER1_Y = Decimal("0.5")
TIER2_Y = Decimal("0.25")
BASE_Y = Decimal("0")
TRUNK_BOTTOM_Y = -TRUNK_H

# Outline, counter-clockwise from the tip. Three stacked tiers plus a trunk.
TREE_VERTICES_DEC: tuple[tuple[Decimal, Decimal], ...] = (
    (Decimal(0), TIP_Y),
    (-TOP_W / 2, TIER1_Y),
    (-TOP_W / 4, TIER1_Y),
    (-MID_W / 2, TIER2_Y),
    (-MID_W / 4, TIER2_Y),
    (-BASE_W / 2, BASE_Y),
    (-TRUNK_W / 2, BASE_Y),
    (-TRUNK_W / 2, TRUNK_BOTTOM_Y),
    (TRUNK_W / 2, TRUNK_BOTTOM_Y),
    (TRUNK_W / 2, BASE_Y),
    (BASE_W / 2, BASE_Y),
    (MID_W / 4, TIER2_Y),
    (MID_W / 2, TIER2_Y),
    (TOP_W / 4, TIER1_Y),
    (TOP_W / 2, TIER1_Y),
)

TREE_VERTICES = np.array([[float(x), float(y)] for x, y in TREE_VERTICES_DEC], dtype=np.float64)

TREE_AREA = 0.245625
# Farthest vertex from the local origin is the tip, so every tree fits in a disc of this radius.
TREE_RADIUS = 0.8


def _f(v: Decimal) -> float:
    return float(v)


# Four convex pieces whose union is the tree: top triangle, middle and bottom
# trapezoids, trunk rectangle. Each is counter-clockwise; the triangle is padded to
# four rows and PIECE_SIZES says how many rows are real.
PIECES = np.array(
    [
        [
            [0.0, _f(TIP_Y)],
            [_f(-TOP_W / 2), _f(TIER1_Y)],
            [_f(TOP_W / 2), _f(TIER1_Y)],
            [_f(TOP_W / 2), _f(TIER1_Y)],
        ],
        [
            [_f(-TOP_W / 4), _f(TIER1_Y)],
            [_f(-MID_W / 2), _f(TIER2_Y)],
            [_f(MID_W / 2), _f(TIER2_Y)],
            [_f(TOP_W / 4), _f(TIER1_Y)],
        ],
        [
            [_f(-MID_W / 4), _f(TIER2_Y)],
            [_f(-BASE_W / 2), _f(BASE_Y)],
            [_f(BASE_W / 2), _f(BASE_Y)],
            [_f(MID_W / 4), _f(TIER2_Y)],
        ],
        [
            [_f(-TRUNK_W / 2), _f(BASE_Y)],
            [_f(-TRUNK_W / 2), _f(TRUNK_BOTTOM_Y)],
            [_f(TRUNK_W / 2), _f(TRUNK_BOTTOM_Y)],
            [_f(TRUNK_W / 2), _f(BASE_Y)],
        ],
    ],
    dtype=np.float64,
)
PIECE_SIZES = np.array([3, 4, 4, 4], dtype=np.int64)


def transform(points: np.ndarray, x: float, y: float, deg: float) -> np.ndarray:
    """Rotate local points by ``deg`` about the origin, then translate by (x, y)."""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T + np.array([x, y])


def tree_polygon(x: float, y: float, deg: float) -> Polygon:
    return Polygon(transform(TREE_VERTICES, x, y, deg))


def piece_polygons(x: float, y: float, deg: float) -> list[Polygon]:
    return [Polygon(transform(PIECES[p, : PIECE_SIZES[p]], x, y, deg)) for p in range(len(PIECES))]
