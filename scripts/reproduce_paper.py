"""Recalcula TODOS os numeros do paper ICDE a partir dos artefatos publicados:
matrizes (data/matrix/), journals online (artifacts/journals/) e resultados do
estudo offline (studies_results/). Nada vem do cluster nem de APIs.

Uso: python -m scripts.reproduce_paper [--out artifacts/paper_numbers.json]

Secoes (mesmos nomes do paper):
  matrix   numeros da matriz (celulas, execucoes, status, horas) e paisagem;
  rq1_rq2  estudo offline (TPC-H 88, TPC-DS 693), media por semente;
  rq3      regret online por SF com a regua top-3 replayada no harness;
  rq3_tpch validacao TPC-H online;
  rq4      ablacao de ancoras no SF100 (media, dp, bootstrap pareado);
  rq5      custo das campanhas online (propostas, falhas, horas, US$);
  repeats  dispersao entre repeticoes (matriz e harness).
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import duckdb

from analysis.oracle import oracle_table
from analysis.report import _load_opt_results, build_report, harness_oracle_table

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "artifacts/journals"
TPCDS = str(ROOT / "data/matrix/training_data.parquet")
TPCH = str(ROOT / "data/matrix/tpch_training_data.parquet")

# Regua top-3 por SF: rank 1 + ranks 2 e 3, mais as re-tentativas de celulas
# que cairam por crash de driver (timeouts ficam como falhas).
RULER = {
    10: ["sf10-replay", "sf10-replay-r2", "sf10-replay-r3", "sf10-replay-r3-rerun"],
    50: ["sf50-replay", "sf50-replay-r2", "sf50-replay-r3", "sf50-replay-r2-rerun", "sf50-replay-r3-rerun"],
    100: ["replay-v2", "sf100-replay-r2", "sf100-replay-r3"],
    250: ["sf250-replay", "sf250-replay-r2", "sf250-replay-r3", "sf250-replay-r2-rerun"],
    500: ["sf500-replay", "sf500-replay-r2", "sf500-replay-r3"],
    750: ["sf750-replay", "sf750-replay-r2", "sf750-replay-r3"],
    1000: ["sf1000-replay", "sf1000-replay-r2", "sf1000-replay-r3"],
}
HYBRID = {10: ["sf10-hybrid3"], 50: ["sf50-hybrid3"], 100: ["hybrid3", "hybrid3-r2"],
          250: ["sf250-hybrid3"], 500: ["sf500-hybrid3"],
          750: ["sf750-hybrid3", "sf750-hybrid3-r2"], 1000: ["sf1000-hybrid3", "sf1000-hybrid3-r2"]}
RANDOM = {10: "sf10-random", 50: "sf50-random", 100: "random-s1", 250: "sf250-random", 500: "sf500-v4-random"}
ANCHORS = {"A0 heuristic": ["sf100-v4-a0-heuristic", "sf100-v4-a0-heuristic-r2", "sf100-v4-a0-heuristic-r3"],
           "A1 small model": ["sf100-v4-a1-jev"],
           "A2 Haiku 4.5": ["sf100-v4-a2-haiku", "sf100-v4-a2-haiku-r2", "sf100-v4-a2-haiku-r3"],
           "A3 Sonnet 5": ["sf100-v4-a3-sonnet"]}
TPCH_RULER = {sf: [f"tpch-sf{sf}-replay", f"tpch-sf{sf}-replay-r2", f"tpch-sf{sf}-replay-r3"] for sf in (500, 750)}
COST = {
    "Hybrid, Sonnet 5 (DS)": ["sf10-hybrid3", "sf50-hybrid3", "hybrid3", "hybrid3-r2", "sf250-hybrid3",
                              "sf500-hybrid3", "sf750-hybrid3", "sf750-hybrid3-r2", "sf1000-hybrid3",
                              "sf1000-hybrid3-r2"],
    "Random (DS)": ["sf10-random", "sf50-random", "random", "random-s1", "sf250-random", "sf500-v4-random"],
    "Anchor ablation (SF100)": ANCHORS["A0 heuristic"] + ANCHORS["A1 small model"] + ANCHORS["A2 Haiku 4.5"]
    + ANCHORS["A3 Sonnet 5"],
    "Anchors (H)": ["tpch-sf500-a2-haiku", "tpch-sf500-a0-heuristic", "tpch-sf750-a2-haiku", "tpch-sf750-a0-heuristic"],
}


def jpath(name):
    return str(J / f"journal-{name}.jsonl")


def rows(run, ruler, matrix=TPCDS):
    rep = build_report(_load_opt_results(jpath(run)), oracle_table(matrix), None,
                       harness_oracle_table([jpath(r) for r in ruler]))
    return {r["query"]: r for r in rep["rows"]}


def med(xs):
    return statistics.median(xs)


def section_matrix():
    c = duckdb.connect()
    n_cells, n_exec, hours = c.execute(f"""SELECT count(DISTINCT (scale_factor, executor_cores, executor_memory_gb,
        workers)), count(*), sum(duration_ms)/3.6e6 FROM read_parquet('{TPCDS}')""").fetchone()
    status = dict(c.execute(f"SELECT status, count(*) FROM read_parquet('{TPCDS}') GROUP BY 1").fetchall())
    c.sql(f"""CREATE VIEW cell AS SELECT CAST(scale_factor AS INT) sf, query, executor_cores, executor_memory_gb,
              workers, min(duration_ms) t FROM read_parquet('{TPCDS}') WHERE status='success' GROUP BY ALL""")
    spread_all = c.execute("SELECT median(r) FROM (SELECT max(t)/min(t) r FROM cell GROUP BY sf, query)").fetchone()[0]
    spread_max = c.execute("SELECT max(r) FROM (SELECT max(t)/min(t) r FROM cell GROUP BY sf, query)").fetchone()[0]
    spread = c.execute("""SELECT sf, median(r), quantile_cont(r, 0.9) FROM
        (SELECT sf, max(t)/min(t) r FROM cell GROUP BY sf, query) GROUP BY sf ORDER BY sf""").fetchall()
    best = c.execute("SELECT sf, query, arg_min(executor_cores || 'c' || workers || 'w', t) FROM cell GROUP BY sf, query").fetchall()
    by = defaultdict(dict)
    for sf, q, s in best:
        by[q][sf] = s
    sfs = sorted({r[0] for r in spread})
    change = {b: round(100 * sum(by[q][a] != by[q][b] for q in by if a in by[q] and b in by[q])
                       / sum(1 for q in by if a in by[q] and b in by[q])) for a, b in zip(sfs, sfs[1:])}
    timeout_cap = c.execute(f"SELECT max(duration_ms)/1000 FROM read_parquet('{TPCDS}') WHERE status='timeout'").fetchone()[0]
    return {"cells": n_cells, "executions": n_exec, "query_hours": round(hours, 1), "status": status,
            "timeout_cap_s": round(timeout_cap), "worst_over_best_median": round(spread_all, 2),
            "worst_over_best_max": round(spread_max, 1),
            "spread_by_sf": {sf: [round(m, 2), round(p, 2)] for sf, m, p in spread},
            "best_shape_change_pct": change}


def section_offline():
    out = {}
    for bench in ("tpch", "tpcds"):
        d = ROOT / "studies_results" / bench
        res = {}
        first = json.load(open(d / "baselines.json"))        # rodada original: fixa e surrogate
        for name, key in (("A0-fixed", "fixed"), ("A2-surrogate", "surrogate")):
            a = first[name]["agg"]
            res[key] = {"seeds": 1, "regret_mean": round(a["regret_mean"], 3), "regret_sd": None,
                        "feasible_pct": round(100 * a["feasibility_hit_rate"], 1)}
        base = json.load(open(d / "baselines_p0.json")) if (d / "baselines_p0.json").exists() else {}
        fam = defaultdict(list)
        for name, v in base.items():
            fam[name.rsplit("-s", 1)[0]].append(v["agg"])
        for f, aggs in fam.items():
            res[f] = {"seeds": len(aggs), "regret_mean": round(statistics.mean(a["regret_mean"] for a in aggs), 3),
                      "regret_sd": round(statistics.stdev([a["regret_mean"] for a in aggs]), 3) if len(aggs) > 1 else None,
                      "feasible_pct": round(100 * statistics.mean(a["feasibility_hit_rate"] for a in aggs), 1)}
        llm, recs = defaultdict(list), defaultdict(list)
        for (arm, it), files in LLM_RUNS[bench].items():
            for f in files:
                (_, v), = json.load(open(d / f)).items()
                llm[f"{arm}-{it}"].append(v["agg"])
                recs[f"{arm}-{it}"].append({(r["dataset"], r["query"]): r for r in v["records"]})
        for k, aggs in llm.items():
            res[k] = {"runs": len(aggs), "regret_mean": round(statistics.mean(a["regret_mean"] for a in aggs), 3),
                      "regret_sd": round(statistics.stdev([a["regret_mean"] for a in aggs]), 3) if len(aggs) > 1 else None,
                      "feasible_pct": round(100 * statistics.mean(a["feasibility_hit_rate"] for a in aggs), 1),
                      "cost_usd": round(sum(a["cost_usd"] for a in aggs), 2)}
        res["telemetry"] = _telemetry_effect(recs["a3m-it4"], recs["a3t-it4"])
        out[bench] = res
    return out


# Rodadas do estudo offline com o MESMO codigo do advisor. A r1 TPC-H de
# setembro (studies_results/tpch/llm_*.json) usou uma versao anterior do
# prompt/contexto (antes de a16db37..54c78aa) e fica fora das medias.
LLM_RUNS = {
    "tpch": {("a3m", "it4"): ["llm_a3m_it4_r2.json", "llm_a3m_it4_r3.json", "llm_a3m_it4_r4.json"],
             ("a3t", "it4"): ["llm_a3t_it4_r2.json", "llm_a3t_it4_r3.json", "llm_a3t_it4_r4.json"],
             ("a4", "it4"): ["llm_a4_it4_r2.json", "llm_a4_it4_r3.json", "llm_a4_it4_r4.json"],
             ("a3m", "it8"): ["llm_a3m_it8_r1.json"], ("a3t", "it8"): ["llm_a3t_it8_r1.json"]},
    "tpcds": {("a3m", "it4"): ["llm_a3m.json", "llm_a3m_it4_r2.json", "llm_a3m_it4_r3.json"],
              ("a3t", "it4"): ["llm_a3t.json", "llm_a3t_it4_r2.json", "llm_a3t_it4_r3.json"],
              ("a4", "it4"): ["llm_a4.json", "llm_a4_it4_r2.json", "llm_a4_it4_r3.json"],
              ("a3m", "it8"): ["llm_a3m_it8_r1.json"], ("a3t", "it8"): ["llm_a3t_it8_r1.json"]},
}


def _telemetry_effect(with_tel, time_only):
    """Diferenca de regret medio (com telemetria - so tempo) por execucao pareada
    (r_i com r_i) e bootstrap pareado por workload sobre todas as execucoes."""
    per_run = []
    pairs = []
    for m, t in zip(with_tel, time_only):
        common = [k for k in m if m[k]["hit"] and k in t and t[k]["hit"]]
        per_run.append(statistics.mean(m[k]["regret"] for k in common) - statistics.mean(t[k]["regret"] for k in common))
        pairs += [m[k]["regret"] - t[k]["regret"] for k in common]
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(pairs, k=len(pairs))) for _ in range(5000))
    rel = statistics.mean(per_run) / statistics.mean(
        statistics.mean(t[k]["regret"] for k in t if t[k]["hit"]) for t in time_only)
    return {"diff_per_run": [round(x, 3) for x in per_run], "diff_mean": round(statistics.mean(per_run), 3),
            "relative": round(rel, 3), "ci95_pooled": [round(bs[125], 3), round(bs[4875], 3)],
            "worse": sum(p > 1e-9 for p in pairs), "better": sum(p < -1e-9 for p in pairs), "pairs": len(pairs)}


def section_rq3():
    out = {}
    for sf, runs in HYBRID.items():
        meds = [med([r["regret_harness"] for r in rows(run, RULER[sf]).values()]) for run in runs]
        medm = [med([r["regret"] for r in rows(run, RULER[sf]).values()]) for run in runs]
        perq = defaultdict(list)
        for run in runs:
            for q, r in rows(run, RULER[sf]).items():
                perq[q].append(r["regret_harness"])
        ruler = {q: round(r["t_star_harness_ms"] / 1000, 1) for q, r in rows(runs[0], RULER[sf]).items()}
        entry = {"hybrid_rho_H_runs": [round(x, 3) for x in meds], "hybrid_rho_H": round(statistics.mean(meds), 2),
                 "hybrid_rho_M": round(statistics.mean(medm), 2),
                 "per_query_mean": {q: round(statistics.mean(v), 3) for q, v in perq.items()}, "ruler_s": ruler}
        if sf in RANDOM:
            rr = rows(RANDOM[sf], RULER[sf]).values()
            entry["random_rho_H"] = round(med([r["regret_harness"] for r in rr]), 2)
            entry["random_rho_M"] = round(med([r["regret"] for r in rr]), 2)
        out[sf] = entry
    # quantas queries tem regua != vencedora da matriz
    flips = 0
    for sf, runs in HYBRID.items():
        first = {r["query"]: r["best_duration_ms"] for r in _load_opt_results(jpath(RULER[sf][0]))}
        for q, r in rows(runs[0], RULER[sf]).items():
            flips += first.get(q) is None or r["t_star_harness_ms"] < first[q] - 1e-6
    out["ruler_not_matrix_winner"] = f"{flips} of {3 * len(HYBRID)}"
    return out


def section_rq3_tpch():
    out = {}
    for sf in (500, 750):
        for arm in ("a2-haiku", "a0-heuristic"):
            rs = rows(f"tpch-sf{sf}-{arm}", TPCH_RULER[sf], TPCH)
            out[f"sf{sf}-{arm}"] = {"median_rho_H": round(med([r["regret_harness"] for r in rs.values()]), 3),
                                    "per_query": {q: round(r["regret_harness"], 3) for q, r in rs.items()}}
    return out


def section_rq4():
    data = {a: [rows(r, RULER[100]) for r in runs] for a, runs in ANCHORS.items()}
    out = {}
    for a, rs in data.items():
        meds = [med([r["regret_harness"] for r in x.values()]) for x in rs]
        out[a] = {"runs": [round(m, 2) for m in meds], "mean": round(statistics.mean(meds), 2),
                  "sd": round(statistics.stdev(meds), 2) if len(meds) > 1 else None}
    a0 = [r["regret_harness"] for x in data["A0 heuristic"] for r in x.values()]
    a2 = [r["regret_harness"] for x in data["A2 Haiku 4.5"] for r in x.values()]
    rng = random.Random(1)
    diffs = sorted(statistics.mean(rng.choices(a2, k=len(a2))) - statistics.mean(rng.choices(a0, k=len(a0)))
                   for _ in range(10000))
    out["A2_minus_A0"] = {"mean": round(statistics.mean(a2) - statistics.mean(a0), 3),
                          "ci95": [round(diffs[250], 3), round(diffs[9750], 3)],
                          "p_A2_better": round(sum(d < 0 for d in diffs) / 10000, 2)}
    return out


def section_rq5():
    out, tot = {}, defaultdict(float)
    for name, runs in COST.items():
        props = fails = 0
        hours = cost = 0.0
        for run in runs:
            recs = [json.loads(l) for l in open(jpath(run))]
            props += sum(r.get("kind") == "opt_iter" for r in recs)
            fails += sum(r.get("kind") == "opt_iter" and r.get("status") != "completed" for r in recs)
            cost += sum(float(r.get("cost_usd") or r.get("cost") or 0) for r in recs if r.get("kind") == "llm_call")
            ts = sorted(r["ts"] for r in recs if r.get("ts"))
            if len(ts) > 1:
                f = lambda x: datetime.fromisoformat(x.replace("Z", "+00:00"))
                hours += (f(ts[-1]) - f(ts[0])).total_seconds() / 3600
        out[name] = {"runs": len(runs), "proposals": props, "failed_pct": round(100 * fails / props),
                     "hours": round(hours, 1), "usd": round(cost, 2)}
        for k, v in (("runs", len(runs)), ("proposals", props), ("fails", fails), ("hours", hours), ("usd", cost)):
            tot[k] += v
    out["Total"] = {"runs": int(tot["runs"]), "proposals": int(tot["proposals"]),
                    "failed_pct": round(100 * tot["fails"] / tot["proposals"]),
                    "hours": round(tot["hours"], 1), "usd": round(tot["usd"], 2)}
    return out


def section_repeats():
    c = duckdb.connect()
    m, share = c.execute(f"""SELECT median(cv), avg(CASE WHEN cv < 0.10 THEN 1 ELSE 0 END) FROM (
        SELECT stddev_samp(duration_ms)/avg(duration_ms) cv FROM read_parquet('{TPCDS}') WHERE status='success'
        GROUP BY scale_factor, query, executor_cores, executor_memory_gb, workers HAVING count(*) >= 3)""").fetchone()
    cvs = []
    for f in glob.glob(str(J / "journal-*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            d = r.get("durations_ms") or []
            if r.get("kind") == "opt_iter" and r.get("status") == "completed" and len(d) >= 3:
                cvs.append(statistics.stdev(d) / statistics.mean(d))
    cvs.sort()
    return {"matrix_cv_median_pct": round(100 * m, 1), "matrix_cv_below10_pct": round(100 * share),
            "harness_cv_median_pct": round(100 * cvs[len(cvs) // 2], 1),
            "harness_cv_below10_pct": round(100 * sum(x < 0.10 for x in cvs) / len(cvs)), "harness_proposals": len(cvs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/paper_numbers.json"))
    a = ap.parse_args()
    res = {"matrix": section_matrix(), "rq1_rq2": section_offline(), "rq3": section_rq3(),
           "rq3_tpch": section_rq3_tpch(), "rq4": section_rq4(), "rq5": section_rq5(),
           "repeats": section_repeats()}
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
