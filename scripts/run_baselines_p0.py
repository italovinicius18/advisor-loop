"""Baselines sem LLM do P0: busca aleatoria e BO com 4 e 8 tentativas, 3 sementes,
em TPC-H e TPC-DS. Saida: studies_results/<bench>/baselines_p0.json.

Uso: python -m scripts.run_baselines_p0
"""
import json
import statistics

from advisor_loop.studies.advisors import BOAdvisor, RandomAdvisor, config_space
from advisor_loop.studies.regret import AdvisorSpec, run_study

REF = {"tpch": "data/matrix/tpch_reference.parquet", "tpcds": "data/matrix/tpcds_reference.parquet"}
SEEDS = (0, 1, 2)

for bench, ref in REF.items():
    space = config_space(ref)
    specs = []
    for k in (4, 8):
        for s in SEEDS:
            specs.append(AdvisorSpec(f"random-{k}-s{s}", RandomAdvisor(space, seed=s), max_iters=k))
            specs.append(AdvisorSpec(f"bo-{k}-s{s}", BOAdvisor(space, seed=s), max_iters=k))
    out = run_study(ref, specs)
    json.dump(out, open(f"studies_results/{bench}/baselines_p0.json", "w"), indent=2)
    print(f"== {bench} ({len(space)} celulas no menu)")
    for fam in ("random-4", "random-8", "bo-4", "bo-8"):
        means = [out[f"{fam}-s{s}"]["agg"]["regret_mean"] for s in SEEDS]
        feas = [out[f"{fam}-s{s}"]["agg"]["feasibility_hit_rate"] for s in SEEDS]
        print(f"   {fam:9s} regret medio por semente {', '.join(f'{m:.3f}' for m in means)} "
              f"-> {statistics.mean(means):.3f} (dp {statistics.stdev(means):.3f}); viaveis {100 * statistics.mean(feas):.1f}%")
