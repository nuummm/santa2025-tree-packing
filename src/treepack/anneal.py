"""Simulated annealing over one layout with hard non-overlap constraints (Numba).

Energy is ``max(w, h) + lam * (w + h)`` of the union box. The first term is the
quantity being scored; the small second term keeps a gradient when several trees sit
on the binding side and moving just one of them would not change ``max(w, h)``.

Only the trees that touch the union box can shrink it, so with probability
``p_boundary`` the moved tree is drawn from that set. Every candidate is first
screened by the Metropolis test (cheap) and then by the conservative collision test,
so the state is always overlap-free. The schedule is a fixed number of moves with
geometric cooling, and the RNG is seeded inside the compiled function, which makes
a run a pure function of (layout, moves, seed, parameters).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .collision import EPS, build_state, collides_with_others, extent, place


@dataclass(frozen=True)
class AnnealParams:
    t0: float = 0.005
    t1: float = 1e-5
    step0: float = 0.05
    step1: float = 1e-3
    rot0: float = 10.0
    rot1: float = 0.1
    p_boundary: float = 0.7
    lam: float = 0.1


@njit(cache=True)
def _energy(mnx, mny, mxx, mxy, lam):
    w = mxx - mnx
    h = mxy - mny
    return max(w, h) + lam * (w + h)


@njit(cache=True)
def _anneal(xs0, ys0, ds0, n_moves, seed, t0, t1, step0, step1, rot0, rot1, p_boundary, lam, eps):
    np.random.seed(seed)
    n = xs0.shape[0]
    xs = xs0.copy()
    ys = ys0.copy()
    ds = ds0.copy()
    v, pb, tb = build_state(xs, ys, ds)
    mnx, mny, mxx, mxy = extent(tb, n)
    e = _energy(mnx, mny, mxx, mxy, lam)
    best_x = xs.copy()
    best_y = ys.copy()
    best_d = ds.copy()
    best_side = max(mxx - mnx, mxy - mny)

    vn = np.empty((v.shape[1], 4, 2))
    pbn = np.empty((v.shape[1], 4))
    tbn = np.empty(4)
    border = np.empty(n, dtype=np.int64)
    log_t = np.log(t1 / t0)
    log_s = np.log(step1 / step0)
    log_r = np.log(rot1 / rot0)
    accepted = 0

    for it in range(n_moves):
        frac = it / n_moves
        temp = t0 * np.exp(frac * log_t)
        step = step0 * np.exp(frac * log_s)
        rot = rot0 * np.exp(frac * log_r)

        i = np.random.randint(n)
        if n > 1 and np.random.random() < p_boundary:
            m = 0
            for k in range(n):
                if tb[k, 0] <= mnx or tb[k, 1] <= mny or tb[k, 2] >= mxx or tb[k, 3] >= mxy:
                    border[m] = k
                    m += 1
            if m > 0:
                i = border[np.random.randint(m)]

        nx = xs[i]
        ny = ys[i]
        nd = ds[i]
        r = np.random.random()
        if r < 0.5:
            nx += step * np.random.normal()
            ny += step * np.random.normal()
        elif r < 0.8:
            nd += rot * np.random.normal()
        else:
            nx += step * np.random.normal()
            ny += step * np.random.normal()
            nd += rot * np.random.normal()

        place(nx, ny, nd, vn, pbn, tbn)
        cmnx = tbn[0]
        cmny = tbn[1]
        cmxx = tbn[2]
        cmxy = tbn[3]
        for k in range(n):
            if k == i:
                continue
            cmnx = min(cmnx, tb[k, 0])
            cmny = min(cmny, tb[k, 1])
            cmxx = max(cmxx, tb[k, 2])
            cmxy = max(cmxy, tb[k, 3])
        e_new = _energy(cmnx, cmny, cmxx, cmxy, lam)
        de = e_new - e
        if de > 0.0 and np.random.random() >= np.exp(-de / temp):
            continue
        if collides_with_others(i, nx, ny, vn, pbn, tbn, xs, ys, v, pb, tb, n, eps):
            continue

        xs[i] = nx
        ys[i] = ny
        ds[i] = nd % 360.0
        v[i] = vn
        pb[i] = pbn
        tb[i] = tbn
        mnx, mny, mxx, mxy = cmnx, cmny, cmxx, cmxy
        e = e_new
        accepted += 1
        side = max(mxx - mnx, mxy - mny)
        if side < best_side:
            best_side = side
            best_x[:] = xs
            best_y[:] = ys
            best_d[:] = ds

    return best_x, best_y, best_d, accepted


def anneal(
    layout: np.ndarray,
    n_moves: int,
    seed: int,
    params: AnnealParams | None = None,
    eps: float = EPS,
) -> tuple[np.ndarray, int]:
    """Anneal an overlap-free layout; returns the best layout seen and the accepted-move count."""
    lay = np.asarray(layout, dtype=np.float64)
    p = params or AnnealParams()
    bx, by, bd, acc = _anneal(
        lay[:, 0].copy(),
        lay[:, 1].copy(),
        lay[:, 2].copy(),
        int(n_moves),
        int(seed),
        p.t0,
        p.t1,
        p.step0,
        p.step1,
        p.rot0,
        p.rot1,
        p.p_boundary,
        p.lam,
        eps,
    )
    return np.column_stack([bx, by, bd]), int(acc)
