"""Turn a full run's results/ into the committed tables benchmarks/ablation.csv and sides.csv.

The stages of one solve are cumulative (each starts from the previous one), so the
per-stage totals form the stacking table. Each stage is reported per seed and as the
mean over seeds; the last row is the per-n ensemble over all seeds.

    uv run treepack solve --moves-per-tree 250000 --seeds 0 1 2
    uv run python scripts/run_ablation.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BENCH = ROOT / "benchmarks"

LABELS = {
    "lattice": "double lattice",
    "anneal": "+ simulated annealing",
    "compact": "+ compaction",
    "remove_one": "+ remove-one cascade",
}


def main() -> None:
    m = json.loads((RESULTS / "metrics.json").read_text())
    seeds = m["config"]["seeds"]
    stages = m["stages"]
    bands = list(stages["ensemble"]["bands"])

    BENCH.mkdir(exist_ok=True)
    with open(BENCH / "ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["stage"]
            + [f"seed{s}" for s in seeds]
            + ["mean", "delta_mean"]
            + [f"n{b.replace('-', '_')}_mean" for b in bands]
        )
        prev = None
        for key, label in LABELS.items():
            runs = [stages[f"seed{s}/{key}"] for s in seeds]
            mean = sum(r["score"] for r in runs) / len(runs)
            band_means = [sum(r["bands"][b] for r in runs) / len(runs) for b in bands]
            delta = "" if prev is None else f"{mean - prev:.4f}"
            w.writerow(
                [label]
                + [f"{r['score']:.4f}" for r in runs]
                + [f"{mean:.4f}", delta]
                + [f"{v:.4f}" for v in band_means]
            )
            prev = mean
        ens = stages["ensemble"]
        w.writerow(
            [f"+ per-n ensemble of seeds {','.join(map(str, seeds))}"]
            + [""] * len(seeds)
            + [f"{ens['score']:.4f}", f"{ens['score'] - prev:.4f}"]
            + [f"{ens['bands'][b]:.4f}" for b in bands]
        )

    with open(RESULTS / "sides.csv") as src:
        table = list(csv.DictReader(src))
    with open(BENCH / "sides.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "lattice", "final"])
        for r in table:
            w.writerow([r["n"], f"{float(r[f'seed{seeds[0]}/lattice']):.6f}", r["ensemble"]])
    print((BENCH / "ablation.csv").read_text())


if __name__ == "__main__":
    main()
