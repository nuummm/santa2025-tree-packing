"""Interlocking double lattice: trees in pairs related by a half turn.

The unit cell holds tree A at (0, 0, theta) and tree B at (u, v, theta + 180), i.e. B
is A turned upside down. Cells repeat along a1 = (a, 0) and a2 = (b, c). For fixed
(theta, u, v) the periods are found by bisection (a for one row, then c for each
shear b). For interlocking shapes feasibility is not monotone in the period, so the
bisection returns a feasible period that is not guaranteed to be the smallest one.
(theta, u, v) are scanned on a grid and the best point is refined by coordinate search.
The figure of merit is the cell area a * c: two trees per a * c of plane.

By translation symmetry a lattice is overlap-free iff A(0) and B(0) are clear of every
translate within reach, so each feasibility check only looks at those two trees.
"""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import asdict, dataclass

import numpy as np
from numba import njit

from .collision import EPS, build_state, layout_side, pair_overlap
from .geometry import TREE_RADIUS

_REACH = 2.0 * TREE_RADIUS + 1e-6


@dataclass(frozen=True)
class LatticeParams:
    theta: float
    u: float
    v: float
    a: float
    b: float
    c: float

    @property
    def cell_area(self) -> float:
        return self.a * self.c

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@njit(cache=True)
def _cell_ok(th, u, v, a, b, c, rows_only, eps):
    tb = th + 180.0
    if pair_overlap(0.0, 0.0, th, u, v, tb, eps):
        return False
    reach = _REACH + np.sqrt(u * u + v * v)
    r_max = 0 if rows_only else int(np.ceil(reach / c)) + 1
    for r in range(-r_max, r_max + 1):
        ox = r * b
        oy = r * c
        k_max = int(np.ceil((reach + abs(ox)) / a)) + 1
        for k in range(-k_max, k_max + 1):
            tx = ox + k * a
            if r != 0 or k != 0:
                if pair_overlap(0.0, 0.0, th, tx, oy, th, eps):
                    return False
                if pair_overlap(u, v, tb, u + tx, v + oy, tb, eps):
                    return False
            if pair_overlap(0.0, 0.0, th, u + tx, v + oy, tb, eps):
                return False
            if pair_overlap(u, v, tb, tx, oy, th, eps):
                return False
    return True


@njit(cache=True)
def _min_a(th, u, v, eps):
    lo = 0.05
    hi = 2.0 * _REACH
    if not _cell_ok(th, u, v, hi, 0.0, 1.0, True, eps):
        return -1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _cell_ok(th, u, v, mid, 0.0, 1.0, True, eps):
            hi = mid
        else:
            lo = mid
    return hi


@njit(cache=True)
def _min_c(th, u, v, a, b, eps):
    lo = 0.05
    hi = 2.0 * (_REACH + abs(v))
    if not _cell_ok(th, u, v, a, b, hi, False, eps):
        return -1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _cell_ok(th, u, v, a, b, mid, False, eps):
            hi = mid
        else:
            lo = mid
    return hi


@njit(cache=True)
def _best_for(th, u, v, n_shear, eps):
    """Smallest cell area over a few row periods and n_shear shears; returns (area, a, b, c)."""
    a0 = _min_a(th, u, v, eps)
    best = (np.inf, 0.0, 0.0, 0.0)
    if a0 < 0:
        return best
    for scale in (1.0, 1.02, 1.05, 1.1):
        a = a0 * scale
        for s in range(n_shear):
            b = a * s / n_shear
            c = _min_c(th, u, v, a, b, eps)
            if c > 0 and a * c < best[0]:
                best = (a * c, a, b, c)
    return best


def _scan_theta(args: tuple[float, np.ndarray, np.ndarray, int, float]) -> tuple:
    th, u_grid, v_grid, n_shear, eps = args
    best = (np.inf, th, 0.0, 0.0, 0.0, 0.0, 0.0)
    for u in u_grid:
        for v in v_grid:
            area, a, b, c = _best_for(th, float(u), float(v), n_shear, eps)
            if area < best[0]:
                best = (area, th, float(u), float(v), a, b, c)
    return best


def search(
    thetas: np.ndarray | None = None,
    u_grid: np.ndarray | None = None,
    v_grid: np.ndarray | None = None,
    n_shear: int = 8,
    refine_iters: int = 40,
    eps: float = EPS,
    pool: Executor | None = None,
) -> LatticeParams:
    """Grid scan over (theta, u, v), then coordinate refinement around the best point.

    theta only needs [0, 90]: theta and theta + 180 give the same lattice (A and B swap)
    and the tree is mirror-symmetric, so -theta gives a mirrored lattice of equal area.
    """
    if thetas is None:
        thetas = np.arange(0.0, 90.0 + 1e-9, 7.5)
    if u_grid is None:
        u_grid = np.linspace(0.0, 1.0, 21)
    if v_grid is None:
        v_grid = np.linspace(-1.0, 1.0, 41)
    jobs = [(float(th), u_grid, v_grid, n_shear, eps) for th in thetas]
    results = list(pool.map(_scan_theta, jobs)) if pool else [_scan_theta(j) for j in jobs]
    best = min(results, key=lambda r: r[0])

    steps = [2.5, float(u_grid[1] - u_grid[0]), float(v_grid[1] - v_grid[0])]
    for _ in range(refine_iters):
        improved = False
        for dim in range(3):
            for sign in (1.0, -1.0):
                th, u, v = best[1], best[2], best[3]
                th, u, v = [
                    (th, u, v)[k] + (sign * steps[k] if k == dim else 0.0) for k in range(3)
                ]
                area, a, b, c = _best_for(th, u, v, n_shear * 2, eps)
                if area < best[0] - 1e-12:
                    best = (area, th, u, v, a, b, c)
                    improved = True
        if not improved:
            steps = [st * 0.5 for st in steps]
    _, th, u, v, a, b, c = best
    return LatticeParams(theta=th, u=u, v=v, a=a, b=b, c=c)


def patch(params: LatticeParams, half_cells: int) -> np.ndarray:
    """Trees of the lattice for cell indices in [-half_cells, half_cells]^2."""
    p = params
    out = []
    for r in range(-half_cells, half_cells + 1):
        for k in range(-half_cells, half_cells + 1):
            x = k * p.a + r * p.b
            y = r * p.c
            out.append((x, y, p.theta % 360.0))
            out.append((x + p.u, y + p.v, (p.theta + 180.0) % 360.0))
    return np.array(out, dtype=np.float64)


def lattice_layout(n: int, params: LatticeParams, n_offsets: int = 16) -> np.ndarray:
    """Cut the n trees that fit in the smallest square window out of the lattice.

    For each window origin (sampled over one cell, since the lattice is periodic) the
    smallest square holding n whole trees is the n-th smallest "reach" of the trees
    that start inside the window.
    """
    p = params
    half = int(np.ceil(np.sqrt(n / 2) * max(p.a, p.c) / min(p.a, p.c))) + 3
    trees = patch(p, half)
    _, _, tb = build_state(trees[:, 0].copy(), trees[:, 1].copy(), trees[:, 2].copy())
    best_idx, best_side = None, np.inf
    for ox in np.arange(n_offsets) * p.a / n_offsets:
        for oy in np.arange(n_offsets) * p.c / n_offsets:
            inside = (tb[:, 0] >= ox) & (tb[:, 1] >= oy)
            reach = np.where(inside, np.maximum(tb[:, 2] - ox, tb[:, 3] - oy), np.inf)
            idx = np.argsort(reach, kind="stable")[:n]
            if not np.isfinite(reach[idx[-1]]):
                continue
            lay = trees[idx]
            side = layout_side(lay[:, 0].copy(), lay[:, 1].copy(), lay[:, 2].copy())
            if side < best_side:
                best_idx, best_side = idx, side
    assert best_idx is not None
    return trees[best_idx].copy()
