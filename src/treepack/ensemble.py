"""Per-n selection across several runs, gated by the final validator."""

from __future__ import annotations

import numpy as np

from .io import layouts_to_rows
from .scoring import side_of
from .validate import validate_rows


def is_valid(n: int, layout: np.ndarray) -> bool:
    rows = layouts_to_rows({n: layout})
    return validate_rows(rows, require_all=False).ok


def ensemble(runs: list[dict[int, np.ndarray]]) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """For every n, the smallest-side layout among ``runs`` that passes validation.

    Returns the chosen layouts and, per n, the index of the run it came from. An n with
    no valid candidate is left out, so the caller's full-file validation will flag it.
    """
    chosen: dict[int, np.ndarray] = {}
    source: dict[int, int] = {}
    for n in sorted(set().union(*runs)):
        cands = sorted(
            ((side_of(run[n]), idx) for idx, run in enumerate(runs) if n in run),
            key=lambda t: (t[0], t[1]),
        )
        for _, idx in cands:
            if is_valid(n, runs[idx][n]):
                chosen[n] = runs[idx][n]
                source[n] = idx
                break
    return chosen, source
