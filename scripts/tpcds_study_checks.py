"""Checagens do estudo offline TPC-DS: telemetria (A3m vs A3t) pareada por workload,
bootstrap, quebra por SF, e onde os bracos erram (hit=False)."""
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

D = Path("studies_results/tpcds")


def recs(f):
    (_, v), = json.load(open(D / f)).items()
    return {(r["dataset"], r["query"]): r for r in v["records"]}


base = json.load(open(D / "baselines.json"))
arms = {"A3m": recs("llm_a3m.json"), "A3t": recs("llm_a3t.json"), "A4": recs("llm_a4.json"),
        "A1": {(r["dataset"], r["query"]): r for r in base["A1-random"]["records"]},
        "A0": {(r["dataset"], r["query"]): r for r in base["A0-fixed"]["records"]}}
m, t = arms["A3m"], arms["A3t"]
pairs = [(m[k]["regret"], t[k]["regret"]) for k in m if m[k]["hit"] and t[k]["hit"]]
diff = [a - b for a, b in pairs]
print(f"pares com ambos viaveis: {len(pairs)}; A3m-A3t media {statistics.mean(diff):+.4f}, mediana {statistics.median(diff):+.4f}")
print(f"  A3m melhor em {sum(d < -1e-9 for d in diff)}, pior em {sum(d > 1e-9 for d in diff)}, empate em {sum(abs(d) <= 1e-9 for d in diff)}")
rng = random.Random(0)
bs = sorted(statistics.mean(rng.choices(diff, k=len(diff))) for _ in range(5000))
print(f"  IC95 bootstrap da diferenca media: [{bs[125]:+.4f}, {bs[4875]:+.4f}]")
print("\npor SF (regret medio, so viaveis) e taxa de acerto:")
sfs = sorted({k[0] for k in m}, key=lambda s: int(s[2:]))
print("SF     " + "  ".join(f"{a:>12s}" for a in arms))
for sf in sfs:
    row = []
    for a, r in arms.items():
        v = [x["regret"] for k, x in r.items() if k[0] == sf and x["hit"]]
        n = sum(1 for k in r if k[0] == sf)
        row.append((f"{statistics.mean(v):.3f}" if v else "  -  ") + f"/{100 * len(v) / n:3.0f}%")
    print(f"{sf:6s} " + "  ".join(f"{c:>12s}" for c in row))
miss = defaultdict(int)
for a, r in arms.items():
    for k, x in r.items():
        if not x["hit"]:
            miss[(a, k[0])] += 1
print("\nfalhas (hit=False) por braco e SF:", dict(sorted(miss.items())))
