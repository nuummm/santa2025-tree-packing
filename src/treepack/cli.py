"""Command line: ``treepack solve`` and ``treepack validate``."""

from __future__ import annotations

import argparse
import os
import sys

from .pipeline import SolveConfig, solve
from .validate import validate_file


def _solve(args: argparse.Namespace) -> int:
    cfg = SolveConfig(
        n_max=args.n_max,
        moves_per_tree=args.moves_per_tree,
        seeds=tuple(args.seeds),
        workers=args.workers,
    )
    m = solve(cfg, args.out)
    print(f"validated score: {float(m['validated_score']):.6f}")
    for name, s in m["stages"].items():
        print(f"  {name:<20} {s['score']:.6f}")
    print(f"wall time: {m['timings']['total_wall_s']:.1f} s -> {args.out}/submission.csv")
    return 0


def _validate(args: argparse.Namespace) -> int:
    report = validate_file(args.file, require_all=not args.partial)
    if not report.ok:
        for e in report.errors:
            print("ERROR", e)
        return 1
    print(f"OK: {len(report.sides)} configurations, score {float(report.score):.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="treepack")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("solve", help="solve n = 1..n-max and write a validated submission")
    p.add_argument("--n-max", type=int, default=200)
    p.add_argument("--moves-per-tree", type=int, default=100_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--out", default="results")
    p.set_defaults(func=_solve)

    p = sub.add_parser("validate", help="check a submission file and print its score")
    p.add_argument("file")
    p.add_argument("--partial", action="store_true", help="do not require all n = 1..200")
    p.set_defaults(func=_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
