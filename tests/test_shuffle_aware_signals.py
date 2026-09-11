"""Utilizacao baixa tem duas causas: CPU ociosa (query leve) ou espera de I/O
(shuffle pesado). Dica e direcao do hibrido devem distinguir as duas."""
from advisor_loop.advisor.advisor import _history_hints
from advisor_loop.advisor.hybrid import HybridAdvisor
from advisor_loop.episode import ConfigVector, Outcome, QueryResult, Recommendation
from advisor_loop.guardian import Guardian

CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}
CFG = dict(executor_cores=4, executor_memory_gb=16, executor_instances=6, driver_memory_gb=8,
           shuffle_partitions=200, advisory_partition_mb=64, aqe_enabled=True,
           broadcast_threshold_mb=10)
GB = 1024 ** 3


def _hist(dur_ms, cpu_ms, shuffle_bytes):
    cv = ConfigVector(**CFG)                                   # 24 cores totais
    out = Outcome(status="completed", per_query={"q": QueryResult(
        status="success", duration_ms=dur_ms,
        metrics={"executor_cpu_time_ms": cpu_ms, "shuffle_write_bytes": shuffle_bytes})})
    return [(cv, out)]


def test_low_util_with_heavy_shuffle_asks_for_more_cores():
    # util = cpu / (24 x 160s) = 10%; shuffle 8.8GB
    h = _history_hints(_hist(160000, int(0.10 * 24 * 160000), int(8.8 * GB))).lower()
    assert "shuffle pesado" in h and "mais cores totais" in h
    assert "cortar cores totais nao deve piorar" not in h


def test_low_util_with_light_shuffle_still_asks_for_fewer_cores():
    h = _history_hints(_hist(40000, int(0.10 * 24 * 40000), 4 * 1024 ** 2)).lower()
    assert "menos cores totais" in h and "shuffle pesado" not in h


def test_hybrid_consolidates_first_when_shuffle_heavy_and_shrinks_when_light():
    """v3: shuffle pesado => 1o passo numerico e CONSOLIDACAO (menos instancias,
    mais cores/exec, cores totais ~iguais); shuffle leve + util baixa => encolhe
    instancias."""
    class LLM:
        def advise(self, *a, **k):
            return Recommendation(rationale="llm", config=ConfigVector(**CFG))
    hy = HybridAdvisor(LLM(), Guardian(CAPS))
    heavy = _hist(160000, int(0.10 * 24 * 160000), int(8.8 * GB))
    light = _hist(40000, int(0.10 * 24 * 40000), 4 * 1024 ** 2)
    r_heavy = hy.advise(None, "c", None, [], history=heavy)
    r_light = hy.advise(None, "c", None, [], history=light)
    assert "consolidacao x2" in r_heavy.rationale and "shuffle-heavy" in r_heavy.rationale
    assert r_heavy.config.executor_instances < 6 and r_heavy.config.executor_cores > 4
    assert abs(r_heavy.config.executor_cores * r_heavy.config.executor_instances - 24) <= 2
    assert r_light.config.executor_instances < 6 and "consolidacao" not in r_light.rationale
