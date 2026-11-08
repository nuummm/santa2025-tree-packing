"""The fast test must be conservative: 'clear' here implies 'clear' for shapely.

Pairs are generated around contact (each pair is pushed until it almost touches), which
is where a wrong SAT axis or a sign error would show up.
"""

import numpy as np
import pytest

from treepack.collision import EPS, count_overlaps, pair_overlap
from treepack.geometry import tree_polygon


def _near_contact_pairs(rng, count):
    """Random pairs whose distance is within ~1e-3 of zero (either side)."""
    pairs = []
    while len(pairs) < count:
        d1, d2 = rng.uniform(0, 360, 2)
        ang = rng.uniform(0, 2 * np.pi)
        ux, uy = np.cos(ang), np.sin(ang)
        lo, hi = 0.0, 1.7
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if tree_polygon(0, 0, d1).intersects(tree_polygon(mid * ux, mid * uy, d2)):
                lo = mid
            else:
                hi = mid
        t = hi + rng.uniform(-1e-3, 1e-3)
        pairs.append((0.0, 0.0, d1, t * ux, t * uy, d2))
    return pairs


@pytest.fixture(scope="module")
def pairs():
    return _near_contact_pairs(np.random.default_rng(0), 300)


def test_known_cases():
    assert pair_overlap(0, 0, 0, 0, 0, 0, EPS)
    assert pair_overlap(0, 0, 0, 0.3, 0, 0, EPS)
    assert not pair_overlap(0, 0, 0, 2.0, 0, 0, EPS)
    assert not pair_overlap(0, 0, 0, 0.71, 0, 0, EPS)
    # Bases meet at a point: shapely calls this touching, the fast test is conservative.
    assert pair_overlap(0, 0, 0, 0.7, 0, 0, EPS)
    assert tree_polygon(0, 0, 0).touches(tree_polygon(0.7, 0, 0))


def test_clear_implies_shapely_clear(pairs):
    for x1, y1, d1, x2, y2, d2 in pairs:
        if not pair_overlap(x1, y1, d1, x2, y2, d2, EPS):
            a, b = tree_polygon(x1, y1, d1), tree_polygon(x2, y2, d2)
            assert a.distance(b) > 0.0


def test_agrees_with_shapely_outside_contact_band(pairs):
    checked = 0
    for x1, y1, d1, x2, y2, d2 in pairs:
        a, b = tree_polygon(x1, y1, d1), tree_polygon(x2, y2, d2)
        fast = pair_overlap(x1, y1, d1, x2, y2, d2, EPS)
        if a.intersection(b).area > 1e-9:
            assert fast
            checked += 1
        elif a.distance(b) > 1e-6:
            assert not fast
            checked += 1
    assert checked > 200


def test_count_overlaps():
    xs = np.array([0.0, 0.2, 5.0])
    ys = np.zeros(3)
    ds = np.zeros(3)
    assert count_overlaps(xs, ys, ds, EPS) == 1
