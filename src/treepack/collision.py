"""Fast, conservative overlap tests for placed trees (Numba).

Each tree is stored as four placed convex pieces plus per-piece and per-tree
axis-aligned boxes. A pair test runs three filters, cheapest first:

1. bounding discs: centres farther apart than 2 * TREE_RADIUS cannot touch;
2. axis-aligned boxes (per tree, then per piece);
3. separating-axis test (SAT) on the edge normals of each convex piece pair.

Every filter only reports "separate" when the gap exceeds ``eps``. Contact or
near-contact therefore counts as a collision, so anything accepted here is
strictly apart and survives the exact final check in :mod:`treepack.validate`.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .geometry import PIECE_SIZES, PIECES, TREE_RADIUS

EPS = 1e-9

_PIECES = PIECES
_SIZES = PIECE_SIZES
_N_PIECES = PIECES.shape[0]
_CUTOFF2 = (2.0 * TREE_RADIUS + 1e-6) ** 2


@njit(cache=True)
def place(x, y, deg, out_v, out_pb, out_tb):
    """Write placed piece vertices, piece boxes and the tree box into the out arrays."""
    t = deg * np.pi / 180.0
    c = np.cos(t)
    s = np.sin(t)
    out_tb[0] = np.inf
    out_tb[1] = np.inf
    out_tb[2] = -np.inf
    out_tb[3] = -np.inf
    for p in range(_N_PIECES):
        mnx = np.inf
        mny = np.inf
        mxx = -np.inf
        mxy = -np.inf
        for k in range(_SIZES[p]):
            vx = _PIECES[p, k, 0]
            vy = _PIECES[p, k, 1]
            wx = c * vx - s * vy + x
            wy = s * vx + c * vy + y
            out_v[p, k, 0] = wx
            out_v[p, k, 1] = wy
            mnx = min(mnx, wx)
            mny = min(mny, wy)
            mxx = max(mxx, wx)
            mxy = max(mxy, wy)
        out_pb[p, 0] = mnx
        out_pb[p, 1] = mny
        out_pb[p, 2] = mxx
        out_pb[p, 3] = mxy
        out_tb[0] = min(out_tb[0], mnx)
        out_tb[1] = min(out_tb[1], mny)
        out_tb[2] = max(out_tb[2], mxx)
        out_tb[3] = max(out_tb[3], mxy)


@njit(cache=True)
def build_state(xs, ys, ds):
    """Placed vertices (n,4,4,2), piece boxes (n,4,4) and tree boxes (n,4) for a layout."""
    n = xs.shape[0]
    v = np.empty((n, _N_PIECES, 4, 2))
    pb = np.empty((n, _N_PIECES, 4))
    tb = np.empty((n, 4))
    for i in range(n):
        place(xs[i], ys[i], ds[i], v[i], pb[i], tb[i])
    return v, pb, tb


@njit(cache=True)
def _has_separating_edge(p, np_, q, nq, eps):
    for i in range(np_):
        j = i + 1 if i + 1 < np_ else 0
        ex = p[j, 0] - p[i, 0]
        ey = p[j, 1] - p[i, 1]
        length = np.sqrt(ex * ex + ey * ey)
        nx = ey / length
        ny = -ex / length
        pmin = np.inf
        pmax = -np.inf
        for k in range(np_):
            d = p[k, 0] * nx + p[k, 1] * ny
            pmin = min(pmin, d)
            pmax = max(pmax, d)
        qmin = np.inf
        qmax = -np.inf
        for k in range(nq):
            d = q[k, 0] * nx + q[k, 1] * ny
            qmin = min(qmin, d)
            qmax = max(qmax, d)
        if qmin - pmax > eps or pmin - qmax > eps:
            return True
    return False


@njit(cache=True)
def convex_overlap(p, np_, q, nq, eps):
    """True unless some edge normal separates the two convex polygons by more than eps."""
    if _has_separating_edge(p, np_, q, nq, eps):
        return False
    return not _has_separating_edge(q, nq, p, np_, eps)


@njit(cache=True)
def _boxes_apart(a, b, eps):
    return a[0] > b[2] + eps or b[0] > a[2] + eps or a[1] > b[3] + eps or b[1] > a[3] + eps


@njit(cache=True)
def trees_overlap(vi, pbi, tbi, vj, pbj, tbj, eps):
    if _boxes_apart(tbi, tbj, eps):
        return False
    for a in range(_N_PIECES):
        for b in range(_N_PIECES):
            if _boxes_apart(pbi[a], pbj[b], eps):
                continue
            if convex_overlap(vi[a], _SIZES[a], vj[b], _SIZES[b], eps):
                return True
    return False


@njit(cache=True)
def pair_overlap(x1, y1, d1, x2, y2, d2, eps):
    """Standalone pair test (places both trees); used by tests and benchmarks."""
    dx = x1 - x2
    dy = y1 - y2
    if dx * dx + dy * dy > _CUTOFF2:
        return False
    v1 = np.empty((_N_PIECES, 4, 2))
    pb1 = np.empty((_N_PIECES, 4))
    tb1 = np.empty(4)
    v2 = np.empty((_N_PIECES, 4, 2))
    pb2 = np.empty((_N_PIECES, 4))
    tb2 = np.empty(4)
    place(x1, y1, d1, v1, pb1, tb1)
    place(x2, y2, d2, v2, pb2, tb2)
    return trees_overlap(v1, pb1, tb1, v2, pb2, tb2, eps)


@njit(cache=True)
def collides_with_others(i, x, y, vi, pbi, tbi, xs, ys, v, pb, tb, n, eps):
    """Does tree ``i`` at its candidate placement hit any other tree j != i (j < n)?"""
    for j in range(n):
        if j == i:
            continue
        dx = x - xs[j]
        dy = y - ys[j]
        if dx * dx + dy * dy > _CUTOFF2:
            continue
        if trees_overlap(vi, pbi, tbi, v[j], pb[j], tb[j], eps):
            return True
    return False


@njit(cache=True)
def count_overlaps(xs, ys, ds, eps):
    """Number of colliding pairs in a layout (O(n^2) with disc and box filters)."""
    n = xs.shape[0]
    v, pb, tb = build_state(xs, ys, ds)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dx + dy * dy > _CUTOFF2:
                continue
            if trees_overlap(v[i], pb[i], tb[i], v[j], pb[j], tb[j], eps):
                count += 1
    return count


@njit(cache=True)
def extent(tb, n):
    """Union box (minx, miny, maxx, maxy) of the first n tree boxes."""
    mnx = np.inf
    mny = np.inf
    mxx = -np.inf
    mxy = -np.inf
    for i in range(n):
        mnx = min(mnx, tb[i, 0])
        mny = min(mny, tb[i, 1])
        mxx = max(mxx, tb[i, 2])
        mxy = max(mxy, tb[i, 3])
    return mnx, mny, mxx, mxy


@njit(cache=True)
def layout_side(xs, ys, ds):
    """Side of the smallest axis-aligned square that contains the layout."""
    _, _, tb = build_state(xs, ys, ds)
    mnx, mny, mxx, mxy = extent(tb, xs.shape[0])
    return max(mxx - mnx, mxy - mny)
