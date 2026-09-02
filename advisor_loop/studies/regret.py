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
class _StudyPack:
    """Contexto justo para o advisor: identifica o workload e o espaço de busca,
    SEM revelar a grade de resultados (isso é o que o loop/histórico realimenta).

    O espaço de configuração é DISCRETO: o menu real de perfis de executor x
    contagem de workers deployáveis (como no LakeHelm). Quando ``menu`` é dado
    (triplas (cores, mem_gb, instances)), o advisor DEVE escolher uma delas —
    caso contrário a proposta não casa com nenhuma célula medida da grade.
    Inofensivo para baselines não-LLM (ignoram o context)."""
    hash = "study"

    def __init__(self, menu: list[tuple[int, int, int]] | None = None):
        self.menu = menu

    def render(self, ep):
        base = (
            f"Workload: query {ep.queries[0]} do TPC-H no dataset {ep.dataset} "
            f"(o sufixo indica o scale factor em GB). Objetivo: minimizar o tempo "
            f"de execução.")
        if self.menu:
            opts = "; ".join(f"({c},{m},{i})" for c, m, i in self.menu)
            return (
                f"{base}\nEscolha EXATAMENTE UM perfil da lista abaixo, no formato "
                f"(executor_cores, executor_memory_gb, executor_instances) — não "
                f"invente valores fora da lista:\n{opts}\nOs demais knobs "
                f"(driver_memory_gb, shuffle_partitions, advisory_partition_mb, "
                f"aqe_enabled, broadcast_threshold_mb) você ajusta livremente. "
                f"Finalize via submit_recommendation.")
        return (
            f"{base}\nVocê escolhe a configuração de execução Spark: executor_cores "
            f"(2-24), executor_memory_gb (4-96), executor_instances (1-20), "
            f"driver_memory_gb, shuffle_partitions, advisory_partition_mb, "
            f"aqe_enabled, broadcast_threshold_mb. Finalize via submit_recommendation.")


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
    pack: object = field(default_factory=_StudyPack)
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
                wls: list[tuple[str, str]], journal=None, pack=None,
                progress=None) -> dict:
    # journal compartilhado com o Advisor LLM => captura o custo real das
    # chamadas de API (baselines não-LLM passam o default de custo zero).
    # pack: contexto do advisor (LLM recebe o menu discreto de configs).
    # progress: callback(idx, total, record, cost_usd) chamado após cada
    # workload -> dá visibilidade e persistência incremental em runs longos.
    journal = journal if journal is not None else _CostJournal()
    deps = _Deps(backend=backend, journal=journal, pack=pack or _StudyPack())
    records = []
    total = len(wls)
    for idx, (dataset, query) in enumerate(wls, start=1):
        ep = Episode(workload="regret-study", dataset=dataset, queries=[query],
                     provider=spec.provider)
        # Um workload que estoure (ex.: o LLM nunca chama submit_recommendation,
        # ou a API falha após esgotar retries) NÃO deve derrubar a varredura
        # inteira: registra como não-hit e segue -> falha honesta e localizada.
        error = None
        try:
            result = optimize(ep, deps, spec.advisor,
                              max_iters=spec.max_iters, patience=spec.patience)
            best = result["best_duration_ms"]
            iters = len(result["iterations"])
        except Exception as e:  # noqa: BLE001 - resiliência de varredura
            best, iters, error = None, 0, f"{type(e).__name__}: {e}"
        orc = backend.oracle(dataset, query)
        regret = (best / orc - 1.0) if (best is not None and orc) else None
        rec = {"dataset": dataset, "query": query,
               "best_ms": best, "oracle_ms": orc, "regret": regret,
               "hit": best is not None, "iters": iters}
        if error:
            rec["error"] = error
        records.append(rec)
        if progress is not None:
            progress(idx, total, rec, journal.cost_usd)
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
