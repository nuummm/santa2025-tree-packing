"""Figures for the README: the n=200 layout, density per n, and the speed comparison.

uv run python scripts/make_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from treepack.geometry import TREE_VERTICES, transform  # noqa: E402
from treepack.io import read_submission  # noqa: E402
from treepack.scoring import density  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SAVE = {"dpi": 160, "bbox_inches": "tight", "metadata": {"Software": None}}


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def layout_figure(layout: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    pts = []
    for x, y, d in layout:
        poly = transform(TREE_VERTICES, x, y, d)
        pts.append(poly)
        # Two shades by which way the tree points, to make the interlocking visible.
        up = np.cos(np.deg2rad(d)) >= 0
        ax.fill(*poly.T, color=BLUE if up else ORANGE, alpha=0.85, linewidth=0.4, ec="white")
    allp = np.vstack(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    side = max(hi - lo)
    ax.add_patch(plt.Rectangle(lo, side, side, fill=False, ec=INK, lw=1.0))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="tip pointing up"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="tip pointing down"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"n = 200, square side {side:.4f}", color=INK)
    fig.savefig(path, **SAVE)
    plt.close(fig)


def density_figure(path: Path) -> None:
    with open(ROOT / "benchmarks" / "sides.csv") as f:
        rows = list(csv.DictReader(f))
    n = np.array([int(r["n"]) for r in rows])
    fin = np.array([density(k, float(r["final"])) for k, r in zip(n, rows, strict=True)])
    lat = np.array([density(k, float(r["lattice"])) for k, r in zip(n, rows, strict=True)])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(n, lat, color=ORANGE, lw=2, label="double lattice only")
    ax.plot(n, fin, color=BLUE, lw=2, label="final (all stages)")
    ax.set_xlabel("n (trees)", color=MUTED)
    ax.set_ylabel("packing density  n·A / side²", color=MUTED)
    ax.set_xlim(1, 200)
    _style(ax)
    ax.legend(frameon=False, labelcolor=INK)
    fig.savefig(path, **SAVE)
    plt.close(fig)


def speed_figure(path: Path) -> None:
    with open(ROOT / "benchmarks" / "collision.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["unit"] == "us/pair"]
    names = [r["method"].replace(", per pair", "") for r in rows]
    vals = [float(r["value"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=BLUE, height=0.55)
    ax.set_yticks(y, names, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("microseconds per tree-pair test (log scale, lower is faster)", color=MUTED)
    for yi, v in zip(y, vals, strict=True):
        ax.text(v * 1.1, yi, f"{v:.3g} µs", va="center", color=INK, fontsize=9)
    ax.invert_yaxis()
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(vals) / 2, max(vals) * 8)
    fig.savefig(path, **SAVE)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    layouts = read_submission(ROOT / "results" / "submission.csv")
    layout_figure(layouts[200], FIG / "n200.png")
    density_figure(FIG / "density_vs_n.png")
    speed_figure(FIG / "speedup.png")


if __name__ == "__main__":
    main()
