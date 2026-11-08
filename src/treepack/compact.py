"""Greedy compaction with a halving step (Numba).

Each sweep first tries to scale the whole layout towards the centre of its union box,
then visits trees from the outside in and tries to move each one towards the centre
(diagonally, then along x, then along y). A move is kept only if it is overlap-free
and does not increase the energy used by the annealer. When a sweep does not lower the
energy, the step is halved; the loop stops below ``step_min``. Because the energy mixes
in ``w + h``, the wrapper returns the input unchanged unless the square side improved.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .anneal import _energy
from .collision import EPS, build_state, collides_with_others, count_overlaps, extent, place
from .scoring import side_of


@njit(cache=True)
def _try_scale(xs, ys, ds, cx, cy, factor, eps):
    nx = cx + (xs - cx) * factor
    ny = cy + (ys - cy) * factor
    if count_overlaps(nx, ny, ds, eps) == 0:
        return True, nx, ny
    return False, xs, ys


@njit(cache=True)
def _compact(xs0, ys0, ds0, step0, step_min, max_sweeps, lam, eps):
    n = xs0.shape[0]
    xs = xs0.copy()
    ys = ys0.copy()
    ds = ds0.copy()
    v, pb, tb = build_state(xs, ys, ds)
    mnx, mny, mxx, mxy = extent(tb, n)
    e = _energy(mnx, mny, mxx, mxy, lam)
    vn = np.empty((v.shape[1], 4, 2))
    pbn = np.empty((v.shape[1], 4))
    tbn = np.empty(4)
    dirs = np.empty((3, 2))
    step = step0
    sweeps = 0
    while step >= step_min and sweeps < max_sweeps:
        sweeps += 1
        e_start = e
        cx = 0.5 * (mnx + mxx)
        cy = 0.5 * (mny + mxy)
        side = max(mxx - mnx, mxy - mny)

        ok, sx, sy = _try_scale(xs, ys, ds, cx, cy, 1.0 - step / side, eps)
        if ok:
            v2, pb2, tb2 = build_state(sx, sy, ds)
            a, b, c, d = extent(tb2, n)
            e2 = _energy(a, b, c, d, lam)
            if e2 <= e:
                xs, ys, v, pb, tb = sx, sy, v2, pb2, tb2
                mnx, mny, mxx, mxy, e = a, b, c, d, e2

        dist = np.maximum(np.abs(xs - cx), np.abs(ys - cy))
        order = np.argsort(-dist)
        for i in order:
            dx = cx - xs[i]
            dy = cy - ys[i]
            length = np.sqrt(dx * dx + dy * dy)
            if length < 1e-12:
                continue
            dirs[0, 0] = dx / length
            dirs[0, 1] = dy / length
            dirs[1, 0] = np.sign(dx)
            dirs[1, 1] = 0.0
            dirs[2, 0] = 0.0
            dirs[2, 1] = np.sign(dy)
            for k in range(3):
                nx = xs[i] + step * dirs[k, 0]
                ny = ys[i] + step * dirs[k, 1]
                place(nx, ny, ds[i], vn, pbn, tbn)
                cmnx = tbn[0]
                cmny = tbn[1]
                cmxx = tbn[2]
                cmxy = tbn[3]
                for j in range(n):
                    if j == i:
                        continue
                    cmnx = min(cmnx, tb[j, 0])
                    cmny = min(cmny, tb[j, 1])
                    cmxx = max(cmxx, tb[j, 2])
                    cmxy = max(cmxy, tb[j, 3])
                e_new = _energy(cmnx, cmny, cmxx, cmxy, lam)
                if e_new > e:
                    continue
                if collides_with_others(i, nx, ny, vn, pbn, tbn, xs, ys, v, pb, tb, n, eps):
                    continue
                xs[i] = nx
                ys[i] = ny
                v[i] = vn
                pb[i] = pbn
                tb[i] = tbn
                mnx, mny, mxx, mxy = cmnx, cmny, cmxx, cmxy
                e = e_new
                break
        if e >= e_start:
            step *= 0.5
    return xs, ys, ds


def compact(
    layout: np.ndarray,
    step0: float = 0.05,
    step_min: float = 1e-6,
    max_sweeps: int = 400,
    lam: float = 0.1,
    eps: float = EPS,
) -> np.ndarray:
    lay = np.asarray(layout, dtype=np.float64)
    xs, ys, ds = _compact(
        lay[:, 0].copy(), lay[:, 1].copy(), lay[:, 2].copy(), step0, step_min, max_sweeps, lam, eps
    )
    out = np.column_stack([xs, ys, ds])
    if side_of(out) < side_of(lay):
        return out
    return lay.copy()
