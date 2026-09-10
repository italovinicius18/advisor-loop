"""Relatorio da campanha Phase-2: regret vs oraculo da matriz, speedup sobre
default, e curva anytime-best.

Metricas (spec secao 2.5):
  regret   = best_ms / t_star - 1   (negativo => bateu o grid)
  speedup  = baseline_ms / best_ms  (>1 => mais rapido que o default)
  anytime  = melhor duracao ate cada iteracao (curva monotona nao-crescente)

build_report cruza as linhas 'opt_result' do journal com o oracle_table e
resume por braco (mediana do regret).
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path


def regret(best_ms: float, t_star: float) -> float:
    return best_ms / t_star - 1.0


def speedup(baseline_ms: float, best_ms: float) -> float:
    return baseline_ms / best_ms


def anytime(iterations: list[dict]) -> list[float]:
    """Melhor (menor) duration_ms observada ate cada iteracao. Ignora None."""
    out: list[float] = []
    best: float | None = None
    for it in iterations:
        d = it.get("duration_ms")
        if d is not None:
            best = d if best is None else min(best, d)
        if best is not None:
            out.append(best)
    return out


def _sf_of(dataset: str) -> int:
    """'sf100' -> 100."""
    return int(str(dataset).lower().removeprefix("sf"))


def build_report(opt_results: list[dict], oracle: dict[tuple[int, str], float],
                 oracle_median: dict[tuple[int, str], float] | None = None) -> dict:
    """`oracle` = t* min (regua historica); `oracle_median` (opcional) = min das
    medianas por celula -> `regret_med`, a regua justa mediana-vs-mediana."""
    oracle_median = oracle_median or {}
    rows = []
    for r in opt_results:
        sf = _sf_of(r["dataset"])
        query = r["query"]
        best = r.get("best_duration_ms")
        base = r.get("baseline_ms")
        t_star = oracle.get((sf, query))
        t_med = oracle_median.get((sf, query))
        rows.append({
            "dataset": r["dataset"], "sf": sf, "query": query,
            "arm": r.get("arm", "llm"),
            "best_duration_ms": best, "baseline_ms": base, "t_star_ms": t_star,
            "t_star_median_ms": t_med,
            "regret": regret(best, t_star) if (best is not None and t_star) else None,
            "regret_med": regret(best, t_med) if (best is not None and t_med) else None,
            "speedup": speedup(base, best) if (base and best) else None,
            "anytime": anytime(r.get("iterations", [])),
        })
    summary: dict[str, dict] = {}
    for arm in sorted({row["arm"] for row in rows}):
        def _med(key):
            vals = [row[key] for row in rows if row["arm"] == arm and row[key] is not None]
            return statistics.median(vals) if vals else None
        summary[arm] = {
            "n": sum(1 for row in rows if row["arm"] == arm),
            "median_regret": _med("regret"),
            "median_regret_med": _med("regret_med"),
            "median_speedup": _med("speedup"),
        }
    return {"rows": rows, "summary": summary}


def _load_opt_results(journal_path: str) -> list[dict]:
    """Le as linhas kind=='opt_result' de um journal jsonl."""
    out = []
    for line in Path(journal_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") == "opt_result":
            out.append(row)
    return out


def main() -> int:
    import argparse
    from analysis.oracle import oracle_table, oracle_table_median
    ap = argparse.ArgumentParser(prog="advisor-report")
    ap.add_argument("journal", help="journal jsonl da campanha")
    ap.add_argument("matrix", help="parquet da matriz (oraculo)")
    ap.add_argument("--out", help="grava o report json aqui", default=None)
    args = ap.parse_args()
    rep = build_report(_load_opt_results(args.journal), oracle_table(args.matrix),
                       oracle_table_median(args.matrix))
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
