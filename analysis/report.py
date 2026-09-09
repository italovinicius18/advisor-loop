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


def build_report(opt_results: list[dict], oracle: dict[tuple[int, str], float]) -> dict:
    rows = []
    for r in opt_results:
        sf = _sf_of(r["dataset"])
        query = r["query"]
        best = r.get("best_duration_ms")
        base = r.get("baseline_ms")
        t_star = oracle.get((sf, query))
        rows.append({
            "dataset": r["dataset"], "sf": sf, "query": query,
            "arm": r.get("arm", "llm"),
            "best_duration_ms": best, "baseline_ms": base, "t_star_ms": t_star,
            "regret": regret(best, t_star) if (best is not None and t_star) else None,
            "speedup": speedup(base, best) if (base and best) else None,
            "anytime": anytime(r.get("iterations", [])),
        })
    summary: dict[str, dict] = {}
    for arm in sorted({row["arm"] for row in rows}):
        regs = [row["regret"] for row in rows if row["arm"] == arm and row["regret"] is not None]
        sps = [row["speedup"] for row in rows if row["arm"] == arm and row["speedup"] is not None]
        summary[arm] = {
            "n": sum(1 for row in rows if row["arm"] == arm),
            "median_regret": statistics.median(regs) if regs else None,
            "median_speedup": statistics.median(sps) if sps else None,
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
    from analysis.oracle import oracle_table
    ap = argparse.ArgumentParser(prog="advisor-report")
    ap.add_argument("journal", help="journal jsonl da campanha")
    ap.add_argument("matrix", help="parquet da matriz (oraculo)")
    ap.add_argument("--out", help="grava o report json aqui", default=None)
    args = ap.parse_args()
    rep = build_report(_load_opt_results(args.journal), oracle_table(args.matrix))
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
