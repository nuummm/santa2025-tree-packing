"""Collision-test and annealing speed: Numba vs straightforward Python/shapely versions.

Writes benchmarks/collision.csv. JIT compilation is triggered before timing, and each
timing is the best of several repeats in a single process.

    uv run python scripts/bench_collision.py
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

from treepack.anneal import anneal
from treepack.collision import EPS, pair_overlap
from treepack.geometry import PIECE_SIZES, PIECES, TREE_VERTICES, transform, tree_polygon
from treepack.lattice import LatticeParams, lattice_layout, search
from treepack.pipeline import machine_info

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "collision.csv"
N_PAIRS = 5_000
REPEATS = 5


def random_pairs(rng: np.random.Generator, count: int) -> np.ndarray:
    """Pairs with centres 0.3-1.6 apart: the range where the full test actually runs."""
    r = rng.uniform(0.3, 1.6, count)
    ang = rng.uniform(0, 2 * np.pi, count)
    d1, d2 = rng.uniform(0, 360, (2, count))
    zeros = np.zeros(count)
    return np.column_stack([zeros, zeros, d1, r * np.cos(ang), r * np.sin(ang), d2])


def best_of(fn, repeats: int = REPEATS) -> float:
    best = math.inf
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


# --- reference implementations -------------------------------------------------------


def shapely_build_and_test(p) -> bool:
    a, b = tree_polygon(p[0], p[1], p[2]), tree_polygon(p[3], p[4], p[5])
    return a.intersects(b) and not a.touches(b)


def _py_place(x, y, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return [
        [(c * vx - s * vy + x, s * vx + c * vy + y) for vx, vy in PIECES[k, : PIECE_SIZES[k]]]
        for k in range(len(PIECES))
    ]


def _py_separated(p, q, eps):
    for i in range(len(p)):
        (x0, y0), (x1, y1) = p[i], p[(i + 1) % len(p)]
        ex, ey = x1 - x0, y1 - y0
        length = math.hypot(ex, ey)
        nx, ny = ey / length, -ex / length
        pp = [vx * nx + vy * ny for vx, vy in p]
        qq = [vx * nx + vy * ny for vx, vy in q]
        if min(qq) - max(pp) > eps or min(pp) - max(qq) > eps:
            return True
    return False


def python_sat(p) -> bool:
    a, b = _py_place(p[0], p[1], p[2]), _py_place(p[3], p[4], p[5])
    return any(
        not (_py_separated(pa, pb, EPS) or _py_separated(pb, pa, EPS)) for pa in a for pb in b
    )


def naive_anneal_moves_per_s(layout: np.ndarray, n_moves: int, seed: int) -> float:
    """Per move: build the moved polygon, test it against all others with shapely's
    vectorised predicates and update the union box. Acceptance is greedy; only the
    per-move cost is being measured."""
    rng = np.random.default_rng(seed)
    lay = layout.copy()
    polys = np.array([tree_polygon(*row) for row in lay], dtype=object)
    bounds = shapely.bounds(polys)
    t = time.perf_counter()
    for _ in range(n_moves):
        i = int(rng.integers(len(lay)))
        x, y, d = lay[i] + [*(0.01 * rng.normal(size=2)), rng.normal()]
        cand = Polygon(transform(TREE_VERTICES, x, y, d))
        others = np.delete(polys, i)
        if np.any(shapely.intersects(cand, others) & ~shapely.touches(cand, others)):
            continue
        new_bounds = bounds.copy()
        new_bounds[i] = cand.bounds
        lo, hi = new_bounds[:, :2].min(axis=0), new_bounds[:, 2:].max(axis=0)
        cur_lo, cur_hi = bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0)
        if np.max(hi - lo) <= np.max(cur_hi - cur_lo):
            lay[i] = (x, y, d)
            polys[i] = cand
            bounds = new_bounds
    return n_moves / (time.perf_counter() - t)


def lattice_from_results() -> LatticeParams:
    """Reuse the lattice of the last full run if present (the full search takes a while)."""
    metrics = Path(__file__).resolve().parents[1] / "results" / "metrics.json"
    if metrics.exists():
        d = json.loads(metrics.read_text())["lattice"]
        return LatticeParams(**{k: d[k] for k in ("theta", "u", "v", "a", "b", "c")})
    return search()


def main() -> None:
    rng = np.random.default_rng(0)
    pairs = random_pairs(rng, N_PAIRS)
    rows = []

    pair_overlap(*pairs[0], EPS)  # JIT warm-up
    t_numba = best_of(lambda: [pair_overlap(*p, EPS) for p in pairs])
    t_pysat = best_of(lambda: [python_sat(p) for p in pairs[:1000]], repeats=3) * N_PAIRS / 1000
    t_shp = best_of(lambda: [shapely_build_and_test(p) for p in pairs[:1000]], repeats=3)
    t_shp *= N_PAIRS / 1000

    fast = np.array([pair_overlap(*p, EPS) for p in pairs])
    ref = np.array([shapely_build_and_test(p) for p in pairs])
    # Conservative by design: the fast test may flag a pair shapely calls clear, never the reverse.
    assert not np.any(ref & ~fast)
    disagree = int(np.sum(fast != ref))

    for name, secs in (
        ("shapely polygons, per pair", t_shp),
        ("pure-Python SAT, per pair", t_pysat),
        ("Numba SAT, per pair", t_numba),
    ):
        rows.append((name, "us/pair", secs / N_PAIRS * 1e6, N_PAIRS))

    lp = lattice_from_results()
    layout = lattice_layout(200, lp)
    anneal(layout, 1000, 0)  # JIT warm-up
    n_moves = 2_000_000
    t_sa = best_of(lambda: anneal(layout, n_moves, 0), repeats=3)
    naive_moves = 2_000
    naive = naive_anneal_moves_per_s(layout, naive_moves, 0)
    rows.append(("shapely annealer, n=200", "moves/s", naive, naive_moves))
    rows.append(("Numba annealer, n=200", "moves/s", n_moves / t_sa, n_moves))

    info = machine_info()
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "unit", "value", "trials", "cpu", "numba"])
        for name, unit, val, trials in rows:
            w.writerow([name, unit, f"{val:.4g}", trials, info["cpu"], info["numba"]])
    for name, unit, val, _ in rows:
        print(f"{name:<32} {val:>12.4g} {unit}")
    print(f"fast-vs-shapely disagreements (all conservative): {disagree}/{N_PAIRS}")


if __name__ == "__main__":
    main()
