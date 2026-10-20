"""Junta os shards do estudo offline (run_llm --shard) e agrega com a mesma funcao
do estudo (advisor_loop.studies.regret.aggregate).

Uso: python scripts/merge_study_shards.py studies_results/tpcds
Escreve <dir>/llm_<arm>.json e <dir>/study_summary.json; imprime a tabela.
"""
import glob
import json
import sys
from pathlib import Path

from advisor_loop.studies.regret import aggregate

D = Path(sys.argv[1])
summary = {}
base = json.load(open(D / "baselines.json"))
for name, v in base.items():
    summary[name] = v["agg"]
for arm in ("a3m", "a3t", "a4"):
    files = sorted(glob.glob(str(D / f"llm_{arm}_*.json")))
    recs, cost, calls, name = [], 0.0, 0, None
    for f in files:
        d = json.load(open(f))
        (name, v), = d.items()
        recs += v["records"]
        cost += v["agg"]["cost_usd"]
        calls += v["agg"]["n_llm_calls"]
    keys = {(r["dataset"], r["query"]) for r in recs}
    assert len(keys) == len(recs), "workload duplicado entre shards"
    agg = aggregate({"records": recs, "cost_usd": cost, "n_llm_calls": calls})
    json.dump({name: {"agg": agg, "records": recs}}, open(D / f"llm_{arm}.json", "w"), indent=2)
    summary[name] = agg
json.dump(summary, open(D / "study_summary.json", "w"), indent=2)
print(f"{'advisor':28s} {'n':>4s} {'hit%':>6s} {'mean':>6s} {'median':>6s} {'p90':>6s} {'US$':>6s}")
for name, a in summary.items():
    print(f"{name:28s} {a['n_workloads']:4d} {100 * a['feasibility_hit_rate']:6.1f} {a['regret_mean']:6.3f} "
          f"{a['regret_median']:6.3f} {a['regret_p90']:6.3f} {a['cost_usd']:6.3f}")
