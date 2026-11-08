"""Starting layouts beyond the lattice: deriving an n-tree layout from an (n+1)-tree one."""

from __future__ import annotations

import numpy as np

from .collision import build_state, extent
from .scoring import side_of


def boundary_trees(layout: np.ndarray) -> np.ndarray:
    """Indices of trees whose box touches the union box."""
    lay = np.asarray(layout, dtype=np.float64)
    _, _, tb = build_state(lay[:, 0].copy(), lay[:, 1].copy(), lay[:, 2].copy())
    mnx, mny, mxx, mxy = extent(tb, len(lay))
    on = (tb[:, 0] <= mnx) | (tb[:, 1] <= mny) | (tb[:, 2] >= mxx) | (tb[:, 3] >= mxy)
    return np.flatnonzero(on)


def remove_one(layout: np.ndarray) -> np.ndarray:
    """Drop the boundary tree whose removal gives the smallest square.

    Only boundary trees can shrink the union box, so interior trees are not tried.
    """
    lay = np.asarray(layout, dtype=np.float64)
    best, best_side = None, np.inf
    for k in boundary_trees(lay):
        cand = np.delete(lay, k, axis=0)
        side = side_of(cand)
        if side < best_side:
            best, best_side = cand, side
    assert best is not None
    return best
