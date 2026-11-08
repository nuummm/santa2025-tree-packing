"""Score bookkeeping: the objective is the sum over n of side(n)^2 / n."""

from __future__ import annotations

import numpy as np

from .collision import layout_side
from .geometry import TREE_AREA

BANDS = ((1, 20), (21, 100), (101, 200))


def side_of(layout: np.ndarray) -> float:
    lay = np.asarray(layout, dtype=np.float64)
    return float(layout_side(lay[:, 0].copy(), lay[:, 1].copy(), lay[:, 2].copy()))


def term(n: int, side: float) -> float:
    return side * side / n


def total_score(sides: dict[int, float]) -> float:
    return float(sum(term(n, s) for n, s in sides.items()))


def density(n: int, side: float) -> float:
    """Fraction of the square covered by trees."""
    return n * TREE_AREA / (side * side)


def band_scores(sides: dict[int, float]) -> dict[str, float]:
    return {
        f"{lo}-{hi}": float(sum(term(n, s) for n, s in sides.items() if lo <= n <= hi))
        for lo, hi in BANDS
    }


def layout_sides(layouts: dict[int, np.ndarray]) -> dict[int, float]:
    return {n: side_of(lay) for n, lay in layouts.items()}
