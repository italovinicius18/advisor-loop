"""Família de advisores-baseline para o estudo de regret offline (Tier 1).

Todos conformam à interface do advisor real:
``advise(ep, context, probe, memories, history=[]) -> Recommendation``,
então ``loop.optimize`` dirige qualquer um deles uniformemente contra o
``ReplayBackend``.

- A0 ``FixedAdvisor``    — config fixa (best-on-average da grade). Sem loop.
- A1 ``RandomAdvisor``   — busca aleatória sobre a grade; usa o histórico
  (embaralhamento determinístico por episódio) para não repetir configs.
- A2 ``SurrogateAdvisor``— zero-shot estilo LakeHelm: transfere, para cada
  query, a melhor config observada em OUTROS scale factors (leave-one-scale-out,
  sem vazamento do SF-alvo). Um único palpite.

A grade só variou (cores, mem, instances); os demais campos do ConfigVector
são defaults fixos (não afetam o lookup do ReplayBackend, que casa só nesses
três eixos).
"""
from __future__ import annotations

import random

import duckdb

from advisor_loop.episode import ConfigVector, Episode, Outcome, Recommendation

# Defaults fixos para os eixos que a grade TPC-H não varreu.
_DEFAULTS = dict(driver_memory_gb=8, shuffle_partitions=200,
                 advisory_partition_mb=64, aqe_enabled=True,
                 broadcast_threshold_mb=10)


def make_config(cores: int, mem_gb: int, instances: int) -> ConfigVector:
    return ConfigVector(executor_cores=cores, executor_memory_gb=mem_gb,
                        executor_instances=instances, **_DEFAULTS)


def config_space(reference) -> list[ConfigVector]:
    """Configs distintas presentes na grade (o menu real de escolhas)."""
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT executor_cores, executor_memory_gb, executor_instances "
        f"FROM read_parquet('{reference}') "
        f"ORDER BY executor_cores, executor_memory_gb, executor_instances").fetchall()
    con.close()
    return [make_config(c, m, i) for c, m, i in rows]


def best_fixed_config(reference, min_feasible_frac: float = 0.9) -> ConfigVector:
    """Config única que minimiza a duração normalizada média (duration/oráculo)
    sobre os workloads onde é viável, exigindo viabilidade em >= min_feasible_frac
    dos workloads. É o baseline estático mais forte (A0)."""
    con = duckdb.connect()
    con.sql(f"CREATE VIEW ref AS SELECT * FROM read_parquet('{reference}')")
    # oráculo por workload
    con.sql("""CREATE VIEW orc AS
        SELECT dataset, query, min(duration_ms) AS best
        FROM ref WHERE status='success' GROUP BY dataset, query""")
    n_workloads = con.execute("SELECT count(*) FROM orc").fetchone()[0]
    rows = con.execute("""
        SELECT r.executor_cores, r.executor_memory_gb, r.executor_instances,
               avg(r.duration_ms / o.best) AS norm, count(*) AS feasible
        FROM ref r JOIN orc o USING (dataset, query)
        WHERE r.status='success'
        GROUP BY 1, 2, 3
        ORDER BY norm""").fetchall()
    con.close()
    for c, m, i, _norm, feasible in rows:
        if feasible >= min_feasible_frac * n_workloads:
            return make_config(c, m, i)
    # fallback: o de menor duração normalizada, mesmo que menos viável
    c, m, i, *_ = rows[0]
    return make_config(c, m, i)


class FixedAdvisor:
    """A0 — sempre a mesma config. Rode com max_iters=1."""

    def __init__(self, config: ConfigVector):
        self.config = config

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list = []) -> Recommendation:
        return Recommendation(config=self.config, rationale="A0 fixed best-on-average")


class RandomAdvisor:
    """A1 — busca aleatória sobre a grade. Embaralhamento determinístico por
    episódio (seed + dataset + query); o índice vem de len(history), então não
    repete config dentro de um episódio e é reproduzível."""

    def __init__(self, space: list[ConfigVector], seed: int = 0):
        self.space = space
        self.seed = seed

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list = []) -> Recommendation:
        rng = random.Random(f"{self.seed}|{ep.dataset}|{ep.queries[0]}")
        order = self.space[:]
        rng.shuffle(order)
        cfg = order[len(history) % len(order)]
        return Recommendation(config=cfg, rationale=f"A1 random trial #{len(history) + 1}")


class FreeRandomAdvisor:
    """Busca aleatoria no SCHEMA LIVRE: amostra cada campo do ConfigVector
    uniformemente na sua faixa, com rejection-sampling contra um Guardian para
    que toda proposta seja viavel. Deterministico por
    (seed, dataset, query, len(history)); evita repetir config dentro do
    episodio. Mesmo espaco de busca do braco LLM -> controle justo (nao a grade
    discreta, como o A1 RandomAdvisor)."""

    _INT = {"executor_cores": (1, 24), "executor_memory_gb": (4, 96),
            "executor_instances": (1, 20), "driver_memory_gb": (4, 16),
            "shuffle_partitions": (100, 4000), "advisory_partition_mb": (16, 256),
            "broadcast_threshold_mb": (-1, 512)}

    def __init__(self, guardian, seed: int = 0, knobs: str = "free"):
        self.guardian = guardian
        self.seed = seed
        # knobs="matrix3": amostra so os 3 eixos da grade e pina os demais
        # (mesma ablacao do braco LLM -> controle random pareado).
        if knobs not in ("free", "matrix3"):
            raise ValueError(f"knobs desconhecido: {knobs!r} (use 'free' ou 'matrix3')")
        self.knobs = knobs

    def _sample(self, rng) -> ConfigVector:
        if self.knobs == "matrix3":
            from advisor_loop.advisor.advisor import PINNED_MATRIX, SHAPE_KNOBS
            f = {k: rng.randint(*self._INT[k]) for k in SHAPE_KNOBS}
            return ConfigVector(**f, **PINNED_MATRIX)
        f = {k: rng.randint(lo, hi) for k, (lo, hi) in self._INT.items()}
        f["aqe_enabled"] = rng.random() < 0.5
        return ConfigVector(**f)

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list = []) -> Recommendation:
        seen = {c.config_hash() for c, _ in history}
        rng = random.Random(f"{self.seed}|{ep.dataset}|{ep.queries[0]}|{len(history)}")
        for _ in range(2000):
            cfg = self._sample(rng)
            if self.guardian.validate(cfg)[0] and cfg.config_hash() not in seen:
                return Recommendation(config=cfg,
                                      rationale=f"free-random trial #{len(history) + 1}")
        raise RuntimeError("FreeRandomAdvisor: sem config viavel e unica em 2000 tentativas")


class SurrogateAdvisor:
    """A2 — proxy zero-shot do LakeHelm: para a query-alvo, a melhor config
    observada em OUTROS scale factors (leave-one-scale-out). Um único palpite;
    rode com max_iters=1. Fallback = best_fixed se a query não aparece fora do
    SF-alvo."""

    def __init__(self, reference):
        self.reference = reference
        self._con = duckdb.connect()
        self._con.sql(f"CREATE VIEW ref AS SELECT * FROM read_parquet('{reference}')")
        self._fallback = best_fixed_config(reference)
        self._cache: dict[str, dict[str, tuple]] = {}

    def _predict(self, dataset: str, query: str) -> ConfigVector:
        if dataset not in self._cache:
            rows = self._con.execute("""
                SELECT query, executor_cores, executor_memory_gb, executor_instances,
                       avg(duration_ms) AS a
                FROM ref WHERE status='success' AND dataset != ?
                GROUP BY 1, 2, 3, 4""", [dataset]).fetchall()
            best: dict[str, tuple] = {}
            for q, c, m, i, a in rows:
                if q not in best or a < best[q][1]:
                    best[q] = ((c, m, i), a)
            self._cache[dataset] = {q: v[0] for q, v in best.items()}
        cfg = self._cache[dataset].get(query)
        return make_config(*cfg) if cfg else self._fallback

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list = []) -> Recommendation:
        cfg = self._predict(ep.dataset, ep.queries[0])
        return Recommendation(config=cfg, rationale="A2 zero-shot cross-scale transfer")
