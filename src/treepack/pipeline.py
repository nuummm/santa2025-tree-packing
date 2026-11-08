"""End-to-end solve: lattice -> anneal -> compact -> remove-one cascade -> ensemble.

Every n is an independent subproblem, so the per-n stages run in a process pool. The
remove-one cascade walks n downwards and is sequential. The budget is a number of moves
(not seconds) and every RNG seed is derived from (seed, n), so a run is reproducible
regardless of the number of workers or the machine's speed.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numba
import numpy as np

from . import __version__
from .anneal import AnnealParams, anneal
from .compact import compact
from .ensemble import ensemble
from .initializers import remove_one
from .io import write_submission
from .lattice import LatticeParams, lattice_layout, search
from .scoring import band_scores, layout_sides, total_score
from .validate import validate_file

STAGES = ("lattice", "anneal", "compact", "remove_one")
MIN_MOVE_TREES = 30


@dataclass
class SolveConfig:
    n_max: int = 200
    moves_per_tree: int = 100_000
    seeds: tuple[int, ...] = (0,)
    workers: int = field(default_factory=lambda: os.cpu_count() or 1)
    anneal: AnnealParams = field(default_factory=AnnealParams)
    lattice: LatticeParams | None = None  # skip the lattice search when given

    def moves(self, n: int) -> int:
        # Small n still get a floor so that the search can rotate and rearrange freely.
        return self.moves_per_tree * max(n, MIN_MOVE_TREES)


def run_seed_for(seed: int, n: int) -> int:
    return seed * 1_000_003 + n


def _solve_n(args: tuple) -> tuple[int, dict[str, np.ndarray], float]:
    n, seed, moves, lp, ap = args
    t = time.perf_counter()
    lay = lattice_layout(n, lp)
    sa, _ = anneal(lay, moves, run_seed_for(seed, n), ap)
    cp = compact(sa)
    return n, {"lattice": lay, "anneal": sa, "compact": cp}, time.perf_counter() - t


def cascade(layouts: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """For n = max-1 .. 1, replace layout n by (layout n+1 minus one tree) if smaller."""
    out = dict(layouts)
    sides = layout_sides(out)
    for n in range(max(out) - 1, 0, -1):
        if n not in out or n + 1 not in out:
            continue
        cand = compact(remove_one(out[n + 1]))
        side = layout_sides({n: cand})[n]
        if side < sides[n]:
            out[n], sides[n] = cand, side
    return out


def run_seed(
    seed: int, cfg: SolveConfig, lp: LatticeParams, pool: ProcessPoolExecutor
) -> tuple[dict[str, dict[int, np.ndarray]], float]:
    stages: dict[str, dict[int, np.ndarray]] = {s: {} for s in STAGES}
    # Largest n first: they are the slowest jobs, so the pool drains evenly.
    jobs = [(n, seed, cfg.moves(n), lp, cfg.anneal) for n in range(cfg.n_max, 0, -1)]
    cpu_seconds = 0.0
    for n, res, secs in pool.map(_solve_n, jobs):
        for name, lay in res.items():
            stages[name][n] = lay
        cpu_seconds += secs
    stages["remove_one"] = cascade(stages["compact"])
    return stages, cpu_seconds


def machine_info() -> dict[str, str | int]:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            pass
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count() or 0,
        "os": platform.system(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "treepack": __version__,
    }


def _summary(sides: dict[int, float]) -> dict:
    return {"score": total_score(sides), "bands": band_scores(sides)}


def solve(cfg: SolveConfig, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        t = time.perf_counter()
        lp = cfg.lattice or search(pool=pool)
        timings["lattice_search_s"] = time.perf_counter() - t
        runs = []
        for seed in cfg.seeds:
            t = time.perf_counter()
            stages, cpu_s = run_seed(seed, cfg, lp, pool)
            timings[f"seed{seed}_wall_s"] = time.perf_counter() - t
            timings[f"seed{seed}_per_n_cpu_s"] = cpu_s
            runs.append(stages)

    final, source = ensemble([r["remove_one"] for r in runs])
    csv_path = out / "submission.csv"
    write_submission(csv_path, final)
    report = validate_file(csv_path, require_all=cfg.n_max == 200)
    timings["total_wall_s"] = time.perf_counter() - t_all
    if not report.ok:
        raise RuntimeError(f"final validation failed: {report.errors[:5]}")

    stage_sides = {
        f"seed{seed}/{name}": layout_sides(run[name])
        for seed, run in zip(cfg.seeds, runs, strict=True)
        for name in STAGES
    }
    stage_sides["ensemble"] = {n: float(s) for n, s in report.sides.items()}
    metrics = {
        "config": {
            "n_max": cfg.n_max,
            "moves_per_tree": cfg.moves_per_tree,
            "min_move_trees": MIN_MOVE_TREES,
            "seeds": list(cfg.seeds),
            "workers": cfg.workers,
            "anneal": asdict(cfg.anneal),
        },
        "lattice": lp.as_dict() | {"cell_area": lp.cell_area},
        "machine": machine_info(),
        "timings": timings,
        "validated_score": str(report.score),
        "stages": {k: _summary(v) for k, v in stage_sides.items()},
        "ensemble_source_counts": {
            str(seed): sum(1 for i in source.values() if i == k) for k, seed in enumerate(cfg.seeds)
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "sides.csv", "w") as f:
        names = list(stage_sides)
        f.write("n," + ",".join(names) + "\n")
        for n in range(1, cfg.n_max + 1):
            f.write(f"{n}," + ",".join(repr(stage_sides[k][n]) for k in names) + "\n")
    return metrics
