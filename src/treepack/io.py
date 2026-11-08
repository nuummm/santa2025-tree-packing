"""Reading and writing submission files.

Rows look like ``001_0,s0.0,s0.0,s45.0``: the id is ``{n:03d}_{index}`` and every value
carries an ``s`` prefix so that it is read back as text, not as a float.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HEADER = ("id", "x", "y", "deg")

Rows = dict[int, list[tuple[str, str, str]]]


def format_value(v: float) -> str:
    # repr() of a Python float is the shortest string that round-trips exactly.
    return "s" + repr(float(v))


def layouts_to_rows(layouts: dict[int, np.ndarray]) -> Rows:
    rows: Rows = {}
    for n, layout in layouts.items():
        if layout.shape != (n, 3):
            raise ValueError(f"layout for n={n} has shape {layout.shape}")
        rows[n] = [(format_value(x), format_value(y), format_value(d)) for x, y, d in layout]
    return rows


def write_submission(path: str | Path, layouts: dict[int, np.ndarray]) -> None:
    rows = layouts_to_rows(layouts)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for n in sorted(rows):
            for i, (x, y, d) in enumerate(rows[n]):
                w.writerow((f"{n:03d}_{i}", x, y, d))


def read_rows(path: str | Path) -> Rows:
    """Raw value strings (``s`` prefix kept) grouped by n, in file order."""
    rows: Rows = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None or tuple(header) != HEADER:
            raise ValueError(f"bad header: {header}")
        for line_no, rec in enumerate(r, start=2):
            if len(rec) != 4:
                raise ValueError(f"line {line_no}: expected 4 fields, got {len(rec)}")
            tid, x, y, d = rec
            group, _, idx = tid.partition("_")
            if not (group.isdigit() and len(group) == 3 and idx.isdigit()):
                raise ValueError(f"line {line_no}: bad id {tid!r}")
            recs = rows.setdefault(int(group), [])
            if int(idx) != len(recs):
                raise ValueError(f"line {line_no}: id {tid!r} out of sequence")
            recs.append((x, y, d))
    return rows


def parse_value(s: str) -> float:
    if not s.startswith("s"):
        raise ValueError(f"value {s!r} lacks the 's' prefix")
    return float(s[1:])


def rows_to_layouts(rows: Rows) -> dict[int, np.ndarray]:
    return {
        n: np.array([[parse_value(v) for v in rec] for rec in recs], dtype=np.float64)
        for n, recs in rows.items()
    }


def read_submission(path: str | Path) -> dict[int, np.ndarray]:
    return rows_to_layouts(read_rows(path))
