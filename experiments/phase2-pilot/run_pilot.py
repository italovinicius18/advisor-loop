"""Phase-2 pilot orchestrator (throwaway spike).

Blind online tuning loop: for each query, measure the Spark default config as a
baseline, then let the LLM propose configs freely (no menu) for N rounds, each a
real local Spark run, feeding back runtime + telemetry + physical plan. Reports
speedup-over-default and the anytime-best curve -- the Phase-2 metrics that
replace grid-relative regret (there is no oracle here).

Run:
  PYTHONPATH=/home/italo/advisor-loop <venv>/bin/python run_pilot.py \
    --data <tpch_sf1_dir> --out results.json --model deepseek/deepseek-v4-flash-0731 \
    --queries q1,q5 --iters 8 --patience 3
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from advisor_loop.advisor.provider import OpenRouterClient

from advisor_blind import advise
from queries import QUERIES

HERE = Path(__file__).resolve().parent
VENV_PY = sys.executable  # invoked with the venv python; reuse it for workers

# Spark stock defaults, expressed as a full ConfigVector, for the baseline run.
DEFAULT_CFG = {
    "executor_cores": 8, "executor_memory_gb": 8, "executor_instances": 1,
    "driver_memory_gb": 4, "shuffle_partitions": 200, "advisory_partition_mb": 64,
    "aqe_enabled": True, "broadcast_threshold_mb": 10,
}


def run_once(data_dir, query_sql, cfg, timeout_s=600):
    """Run one query in a fresh Spark process; return its result dict."""
    out = HERE / "_run.json"
    if out.exists():
        out.unlink()
    proc = subprocess.run(
        [VENV_PY, str(HERE / "spark_run.py"), data_dir, query_sql,
         json.dumps(cfg), str(out)],
        capture_output=True, text=True, timeout=timeout_s,
    )
    if not out.exists():
        return {"status": "failed", "duration_ms": None, "metrics": {},
                "plan": "", "error": (proc.stderr or "no output")[-800:]}
    return json.loads(out.read_text())


def tune_query(client, model, data_dir, qname, iters, patience):
    qsql = QUERIES[qname]
    print(f"\n=== {qname}: baseline (default config) ===", flush=True)
    base = run_once(data_dir, qsql, DEFAULT_CFG)
    if base["status"] != "success":
        print(f"  baseline failed: {base.get('error')}", flush=True)
        return {"query": qname, "baseline": base, "iterations": [], "error": "baseline_failed"}
    baseline_ms = base["duration_ms"]
    default_plan = base["plan"]
    print(f"  baseline = {baseline_ms:.0f} ms", flush=True)

    history, iterations = [], []
    best_ms, best_cfg = baseline_ms, dict(DEFAULT_CFG)
    no_improve, cost = 0, 0.0
    for i in range(1, iters + 1):
        cfg_obj, rationale, c = advise(client, model, qname, qsql, default_plan, history)
        cost += c
        cfg = cfg_obj.model_dump()
        res = run_once(data_dir, qsql, cfg)
        rec = {"config": cfg, "status": res["status"],
               "duration_ms": res.get("duration_ms"),
               "metrics": res.get("metrics", {}), "rationale": rationale,
               "error": res.get("error", "")}
        history.append(rec)
        dur = res.get("duration_ms")
        improved = dur is not None and dur < best_ms
        if improved:
            best_ms, best_cfg, no_improve = dur, cfg, 0
        else:
            no_improve += 1
        tag = f"{dur:.0f} ms" if dur is not None else res["status"].upper()
        star = " *best" if improved else ""
        iterations.append({"iter": i, **rec, "best_ms_so_far": best_ms})
        print(f"  iter {i}: {tag}{star}  (best {best_ms:.0f} ms, "
              f"{baseline_ms/best_ms:.2f}x) — {rationale[:80]}", flush=True)
        if no_improve >= patience:
            print(f"  stop: {patience} rounds without improvement", flush=True)
            break

    return {
        "query": qname, "baseline_ms": baseline_ms, "default_plan": default_plan,
        "best_ms": best_ms, "best_config": best_cfg,
        "speedup_vs_default": baseline_ms / best_ms,
        "llm_cost_usd": cost, "iterations": iterations,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--queries", default="q1,q5")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    client = OpenRouterClient(timeout=90)
    t0 = time.time()
    results = []
    for qname in args.queries.split(","):
        results.append(tune_query(client, args.model, args.data, qname,
                                  args.iters, args.patience))
        Path(args.out).write_text(json.dumps(results, indent=2))  # incremental

    total_cost = sum(r.get("llm_cost_usd", 0) for r in results)
    print(f"\n=== done in {(time.time()-t0)/60:.1f} min, LLM cost ${total_cost:.4f} ===",
          flush=True)
    for r in results:
        if "speedup_vs_default" in r:
            print(f"  {r['query']}: {r['baseline_ms']:.0f} -> {r['best_ms']:.0f} ms "
                  f"({r['speedup_vs_default']:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
