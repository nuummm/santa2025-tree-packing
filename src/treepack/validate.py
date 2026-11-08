"""Final gate: re-check a submission from its text with exact-decimal inputs and shapely.

This is independent of the Numba code used during search. Values are parsed from the
strings as ``Decimal``, scaled up by a large constant before being handed to shapely
(coordinates far from 1 keep shapely's absolute tolerances negligible), and two trees
are reported as overlapping when their polygons intersect without merely touching.
Every n present in the file is checked, not a sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from .geometry import TREE_VERTICES_DEC
from .io import Rows, read_rows

SCALE = Decimal("1e18")
COORD_LIMIT = Decimal(100)
N_MAX = 200

_BASE = Polygon([(float(x * SCALE), float(y * SCALE)) for x, y in TREE_VERTICES_DEC])


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    sides: dict[int, Decimal] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def score(self) -> Decimal:
        return sum((s * s / n for n, s in self.sides.items()), Decimal(0))


def _to_decimal(s: str) -> Decimal:
    if not s.startswith("s"):
        raise ValueError(f"value {s!r} lacks the 's' prefix")
    return Decimal(s[1:])


def scaled_polygon(x: Decimal, y: Decimal, deg: Decimal) -> Polygon:
    rotated = affinity.rotate(_BASE, float(deg), origin=(0, 0))
    return affinity.translate(rotated, xoff=float(x * SCALE), yoff=float(y * SCALE))


def overlapping_pairs(polys: list[Polygon]) -> list[tuple[int, int]]:
    tree = STRtree(polys)
    pairs = []
    for i, p in enumerate(polys):
        for j in tree.query(p):
            j = int(j)
            if j > i and p.intersects(polys[j]) and not p.touches(polys[j]):
                pairs.append((i, j))
    return pairs


def check_group(n: int, recs: list[tuple[str, str, str]], report: Report) -> None:
    if len(recs) != n:
        report.errors.append(f"n={n}: expected {n} rows, found {len(recs)}")
        return
    try:
        vals = [tuple(_to_decimal(v) for v in rec) for rec in recs]
    except (ValueError, InvalidOperation) as e:
        report.errors.append(f"n={n}: {e}")
        return
    for i, (x, y, _) in enumerate(vals):
        if not (-COORD_LIMIT <= x <= COORD_LIMIT and -COORD_LIMIT <= y <= COORD_LIMIT):
            report.errors.append(f"n={n}: tree {i} outside [-100, 100]")
            return
    polys = [scaled_polygon(*v) for v in vals]
    pairs = overlapping_pairs(polys)
    if pairs:
        report.errors.append(f"n={n}: {len(pairs)} overlapping pair(s), first {pairs[0]}")
        return
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    report.sides[n] = Decimal(max(maxx - minx, maxy - miny)) / SCALE


def validate_rows(rows: Rows, require_all: bool = True) -> Report:
    report = Report()
    expected = range(1, N_MAX + 1)
    if require_all:
        missing = [n for n in expected if n not in rows]
        if missing:
            report.errors.append(f"missing n: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    for n in sorted(rows):
        if not 1 <= n <= N_MAX:
            report.errors.append(f"unexpected n={n}")
            continue
        check_group(n, rows[n], report)
    return report


def validate_file(path: str | Path, require_all: bool = True) -> Report:
    try:
        rows = read_rows(path)
    except ValueError as e:
        return Report(errors=[str(e)])
    return validate_rows(rows, require_all=require_all)
