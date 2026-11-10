"""Junta os shards do estudo offline (run_llm --shard) e agrega com a mesma funcao
do estudo (advisor_loop.studies.regret.aggregate).

Agrupa por prefixo: llm_<arm>[_it<k>_<rep>]_<i>of<n>.json -> llm_<arm>[_it<k>_<rep>].json.
Os shards da primeira rodada (llm_a3m_0of3.json) viram llm_a3m.json, como antes.

Uso: python -m scripts.merge_study_shards studies_results/tpcds
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from advisor_loop.studies.regret import aggregate

D = Path(sys.argv[1])
groups = defaultdict(list)
for f in sorted(glob.glob(str(D / "llm_*.json"))):
    m = re.match(r"(llm_.+?)_(\d+of\d+|all)\.json$", Path(f).name)
    if m:
        groups[m.group(1)].append(f)
for prefix, files in sorted(groups.items()):
    recs, cost, calls, name = [], 0.0, 0, None
    for f in files:
        (name, v), = json.load(open(f)).items()
        recs += v["records"]
        cost += v["agg"]["cost_usd"]
        calls += v["agg"]["n_llm_calls"]
    keys = {(r["dataset"], r["query"]) for r in recs}
    assert len(keys) == len(recs), f"workload duplicado entre shards de {prefix}"
    agg = aggregate({"records": recs, "cost_usd": cost, "n_llm_calls": calls})
    json.dump({name: {"agg": agg, "records": recs}}, open(D / f"{prefix}.json", "w"), indent=2)
    print(f"{prefix:28s} shards={len(files)} n={agg['n_workloads']} viaveis={100 * agg['feasibility_hit_rate']:.1f}% "
          f"regret={agg['regret_mean']:.3f} US$={agg['cost_usd']:.3f}")
