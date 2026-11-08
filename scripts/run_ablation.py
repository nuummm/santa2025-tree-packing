"""Turn a full run's results/ into the committed tables benchmarks/ablation.csv and sides.csv.

The stages of one solve are cumulative (each starts from the previous one), so the
per-stage totals of seed 0 form the stacking table; the last row adds the per-n ensemble
over all seeds.

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
    rows = [(LABELS[k], f"seed{seeds[0]}/{k}") for k in LABELS]
    rows.append((f"+ per-n ensemble of seeds {','.join(map(str, seeds))}", "ensemble"))

    BENCH.mkdir(exist_ok=True)
    with open(BENCH / "ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "score", "delta", "n1_20", "n21_100", "n101_200"])
        prev = None
        for label, key in rows:
            s = stages[key]
            delta = "" if prev is None else f"{s['score'] - prev:.4f}"
            b = s["bands"]
            w.writerow([label, f"{s['score']:.4f}", delta] + [f"{b[k]:.4f}" for k in b])
            prev = s["score"]
        for seed in seeds:
            s = stages[f"seed{seed}/remove_one"]
            w.writerow([f"seed {seed} alone (all stages)", f"{s['score']:.4f}", "", "", "", ""])

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
