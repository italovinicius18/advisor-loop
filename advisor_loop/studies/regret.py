"""Runner do estudo de regret offline (Tier 1).

Para cada advisor e cada workload ``(dataset, query)``, roda o laço fechado
``loop.optimize`` contra o ``ReplayBackend`` (sem cluster) e mede o regret vs
o oráculo da grade:

    regret(A, W) = t_A(W) / t*(W) - 1      (t* = melhor config da grade)

Agrega por advisor: taxa de viabilidade (achou >=1 sucesso), regret
mean/median/p90 sobre os hits, custo LLM (0 nos baselines não-LLM).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import duckdb

from advisor_loop.backends.replay import ReplayBackend
from advisor_loop.episode import Episode
from advisor_loop.loop import optimize


# --- deps mínimos para dirigir loop.optimize offline ---------------------
class _NullPack:
    hash = "null"

    def render(self, ep):
        return ""


class _NullMemory:
    def retrieve(self, ep):
        return []

    def distill(self, ep, rec, out):
        pass


class _PassGuardian:
    def validate(self, config):
        return True, ""


class _CostJournal:
    """Soma o custo real das chamadas LLM (0 para baselines não-LLM)."""

    def __init__(self):
        self.cost_usd = 0.0
        self.n_llm = 0

    def append(self, e):
        if e.get("kind") == "llm_call":
            self.cost_usd += e.get("cost_usd", 0.0) or 0.0
            self.n_llm += 1

    def has(self, k):
        return False


@dataclass
class _Deps:
    backend: object
    journal: object
    pack: object = field(default_factory=_NullPack)
    guardian: object = field(default_factory=_PassGuardian)
    memory: object = field(default_factory=_NullMemory)
    probe_factory: object = field(default=lambda ep: None)


@dataclass
class AdvisorSpec:
    name: str
    advisor: object
    max_iters: int = 8
    patience: int = 999          # baselines não devem parar cedo
    provider: str = "offline"    # model string real só para os braços LLM


def workloads(reference) -> list[tuple[str, str]]:
    """Workloads com oráculo (>=1 config success)."""
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT dataset, query FROM read_parquet('{reference}') "
        f"WHERE status='success' ORDER BY dataset, query").fetchall()
    con.close()
    return [(d, q) for d, q in rows]


def _p90(vals: list[float]) -> float | None:
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    return statistics.quantiles(vals, n=10, method="inclusive")[8]


def run_advisor(spec: AdvisorSpec, backend: ReplayBackend,
                wls: list[tuple[str, str]]) -> dict:
    journal = _CostJournal()
    deps = _Deps(backend=backend, journal=journal)
    records = []
    for dataset, query in wls:
        ep = Episode(workload="regret-study", dataset=dataset, queries=[query],
                     provider=spec.provider)
        result = optimize(ep, deps, spec.advisor,
                          max_iters=spec.max_iters, patience=spec.patience)
        best = result["best_duration_ms"]
        orc = backend.oracle(dataset, query)
        regret = (best / orc - 1.0) if (best is not None and orc) else None
        records.append({"dataset": dataset, "query": query,
                        "best_ms": best, "oracle_ms": orc, "regret": regret,
                        "hit": best is not None,
                        "iters": len(result["iterations"])})
    return {"records": records, "cost_usd": journal.cost_usd,
            "n_llm_calls": journal.n_llm}


def aggregate(run: dict) -> dict:
    records = run["records"]
    n = len(records)
    hits = [r for r in records if r["hit"]]
    regrets = sorted(r["regret"] for r in hits)
    return {
        "n_workloads": n,
        "feasibility_hit_rate": (len(hits) / n) if n else 0.0,
        "regret_mean": (sum(regrets) / len(regrets)) if regrets else None,
        "regret_median": statistics.median(regrets) if regrets else None,
        "regret_p90": _p90(regrets),
        "cost_usd": run["cost_usd"],
        "n_llm_calls": run["n_llm_calls"],
    }


def run_study(reference, specs: list[AdvisorSpec]) -> dict:
    backend = ReplayBackend(reference)
    wls = workloads(reference)
    out = {}
    for spec in specs:
        run = run_advisor(spec, backend, wls)
        out[spec.name] = {"agg": aggregate(run), "records": run["records"]}
    return out
