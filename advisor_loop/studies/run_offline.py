"""CLI do estudo de regret offline (Tier 1).

    python -m advisor_loop.studies.run_offline REFERENCE.parquet [--out results.json]
                                               [--random-iters N] [--seed S]

Roda os baselines não-LLM A0 (fixed best-on-average), A1 (random search) e
A2 (surrogate zero-shot cross-scale) — custo zero, sem tocar o cluster.
Os braços LLM A3/A4 ficam num runner separado (custam $ via API).
"""
from __future__ import annotations

import argparse
import json
import sys

from advisor_loop.studies.advisors import (
    config_space, best_fixed_config, FixedAdvisor, RandomAdvisor, SurrogateAdvisor,
)
from advisor_loop.studies.regret import AdvisorSpec, run_study


def build_baseline_specs(reference, random_iters: int = 8, seed: int = 0) -> list[AdvisorSpec]:
    space = config_space(reference)
    return [
        AdvisorSpec("A0-fixed", FixedAdvisor(best_fixed_config(reference)), max_iters=1),
        AdvisorSpec("A1-random", RandomAdvisor(space, seed=seed), max_iters=random_iters),
        AdvisorSpec("A2-surrogate", SurrogateAdvisor(reference), max_iters=1),
    ]


def _fmt(v, pct=False):
    if v is None:
        return "  n/a"
    return f"{v * 100:5.1f}%" if pct else f"{v:6.3f}"


def print_table(results: dict) -> None:
    print(f"{'advisor':<14} {'hit':>7} {'regret_mean':>12} "
          f"{'median':>8} {'p90':>8} {'cost$':>8}")
    print("-" * 62)
    for name, r in results.items():
        a = r["agg"]
        print(f"{name:<14} {_fmt(a['feasibility_hit_rate'], pct=True):>7} "
              f"{_fmt(a['regret_mean']):>12} {_fmt(a['regret_median']):>8} "
              f"{_fmt(a['regret_p90']):>8} {a['cost_usd']:8.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="run_offline")
    ap.add_argument("reference")
    ap.add_argument("--out")
    ap.add_argument("--random-iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    specs = build_baseline_specs(args.reference, args.random_iters, args.seed)
    results = run_study(args.reference, specs)
    n = next(iter(results.values()))["agg"]["n_workloads"]
    print(f"workloads (SF,query) com oráculo: {n}\n")
    print_table(results)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nresultados: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
