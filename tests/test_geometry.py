import numpy as np
import pytest
from shapely.ops import unary_union

from treepack.geometry import (
    TREE_AREA,
    TREE_RADIUS,
    TREE_VERTICES,
    piece_polygons,
    transform,
    tree_polygon,
)
from treepack.scoring import side_of


def test_area_matches_constant():
    assert tree_polygon(0, 0, 0).area == pytest.approx(TREE_AREA, abs=1e-12)


def test_polygon_is_simple():
    assert tree_polygon(0, 0, 0).is_valid
    assert len(TREE_VERTICES) == 15


def test_pieces_are_convex_and_union_is_tree():
    pieces = piece_polygons(0.3, -1.2, 37.0)
    for p in pieces:
        assert p.convex_hull.area == pytest.approx(p.area, abs=1e-12)
    assert sum(p.area for p in pieces) == pytest.approx(TREE_AREA, abs=1e-12)
    assert unary_union(pieces).symmetric_difference(tree_polygon(0.3, -1.2, 37.0)).area < 1e-12


def test_radius_is_tip_distance():
    assert np.max(np.hypot(TREE_VERTICES[:, 0], TREE_VERTICES[:, 1])) == pytest.approx(TREE_RADIUS)


def test_single_upright_tree_has_unit_side():
    assert side_of(np.array([[0.0, 0.0, 0.0]])) == pytest.approx(1.0)


def test_full_turn_is_identity():
    np.testing.assert_allclose(
        transform(TREE_VERTICES, 1, 2, 360.0) - [1, 2], TREE_VERTICES, atol=1e-12
    )


def test_rotation_is_counter_clockwise():
    tip = transform(np.array([[0.0, 0.8]]), 0, 0, 90.0)[0]
    np.testing.assert_allclose(tip, [-0.8, 0.0], atol=1e-12)
