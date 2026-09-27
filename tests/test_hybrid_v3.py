"""Hibrido v3: consolidacao primeiro em shuffle-heavy; LLM repetido cai no numerico."""
from advisor_loop.advisor.hybrid import HybridAdvisor, consolidate
from advisor_loop.episode import ConfigVector, Outcome, QueryResult, Recommendation
from advisor_loop.guardian import Guardian

CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}
CFG = dict(executor_cores=5, executor_memory_gb=12, executor_instances=16, driver_memory_gb=8,
           shuffle_partitions=200, advisory_partition_mb=64, aqe_enabled=True,
           broadcast_threshold_mb=10)
GB = 1024 ** 3


def _hist(cfg, dur_ms, cpu_ms, shuffle):
    out = Outcome(status="completed", per_query={"q": QueryResult(
        status="success", duration_ms=dur_ms,
        metrics={"executor_cpu_time_ms": cpu_ms, "shuffle_write_bytes": shuffle})})
    return (ConfigVector(**cfg), out)


class LLM:
    def __init__(self, cfg):
        self.cfg, self.calls = cfg, 0
    def advise(self, *a, **k):
        self.calls += 1
        return Recommendation(rationale="llm", config=ConfigVector(**self.cfg))


def test_consolidate_keeps_total_cores_and_mem_per_core():
    cv = ConfigVector(**CFG)                        # 5c x 16w x 12g = 80c, 2.4 g/core
    c2 = consolidate(cv, 2.0)
    assert (c2.executor_cores, c2.executor_instances, c2.executor_memory_gb) == (10, 8, 24)
    c4 = consolidate(cv, 4.0)
    assert (c4.executor_cores, c4.executor_instances, c4.executor_memory_gb) == (20, 4, 48)
    half = consolidate(cv, 0.5)                     # 2-3c x 20w (clamp 20 instancias)
    assert half.executor_instances == 20 and half.executor_cores < 5
    assert c2.shuffle_partitions == cv.shuffle_partitions   # knobs extras preservados


def test_shuffle_heavy_first_numeric_step_is_consolidation():
    hy = HybridAdvisor(LLM(CFG), Guardian(CAPS))
    hist = [_hist(CFG, 170000, int(0.14 * 80 * 170000), int(10 * GB))]
    r = hy.advise(None, "c", None, [], history=hist)
    assert "consolidacao x2" in r.rationale
    assert (r.config.executor_cores, r.config.executor_instances) == (10, 8)
    hist.append((r.config, hist[0][1]))             # x2 visto (nao melhorou: mesmo dur)
    r2 = hy.advise(None, "c", None, [], history=hist)
    assert "consolidacao x4" in r2.rationale and r2.config.executor_cores == 20


def test_consolidation_capped_at_64g_and_2_instances():
    """v3.1: 24c x 1w x 96g e 24c x 2w x 96g crasharam o driver na rodada v3;
    a consolidacao nao pode propor >64g nem <2 instancias."""
    hy = HybridAdvisor(LLM(CFG), Guardian(CAPS))
    big = {**CFG, "executor_cores": 12, "executor_instances": 4, "executor_memory_gb": 48}
    hist = [_hist(big, 120000, int(0.3 * 48 * 120000), int(10 * GB))]
    r = hy.advise(None, "c", None, [], history=hist)
    # x2 daria 24c x 2w x 96g (>64g) -> pulado; x4 daria 1 instancia -> pulado;
    # x0.5 (6c x 8w x 24g) e o proximo candidato de consolidacao
    assert "consolidacao x0.5" in r.rationale
    assert r.config.executor_memory_gb <= 64 and r.config.executor_instances >= 2


def test_light_shuffle_does_not_consolidate_first():
    hy = HybridAdvisor(LLM(CFG), Guardian(CAPS))
    hist = [_hist(CFG, 20000, int(0.3 * 80 * 20000), 4 * 1024 ** 2)]
    r = hy.advise(None, "c", None, [], history=hist)
    assert "consolidacao" not in r.rationale and "executor_instances" in r.rationale


def test_llm_repeating_seen_config_falls_through_to_numeric():
    llm = LLM(CFG)
    hy = HybridAdvisor(llm, Guardian(CAPS), llm_every=3)
    h0 = _hist(CFG, 170000, 1000, int(10 * GB))
    hist = [h0, (consolidate(h0[0], 2.0), h0[1]), (consolidate(h0[0], 4.0), h0[1])]
    r = hy.advise(None, "c", None, [], history=hist)   # n=3 -> LLM, que repete CFG
    assert llm.calls == 1
    assert r.config.config_hash() != ConfigVector(**CFG).config_hash()
    assert r.rationale.startswith("LLM repetiu config ja vista")


def _fail(cfg):
    return (ConfigVector(**cfg), Outcome(status="completed", per_query={"q": QueryResult(
        status="failed", duration_ms=None, metrics={})}))


def test_does_not_retry_failed_shape_with_less_or_equal_memory():
    """sf750 q23 (25/09): 4c x 20w x 16g abortou no shuffle e o x0.5 seguinte
    propos 4c x 20w x 14g -- mesmo formato, menos memoria, falhou de novo (~50 min
    cada). Formato (cores, instancias) que falhou com memoria >= a da candidata
    nao e reproposto."""
    best = {**CFG, "executor_cores": 8, "executor_instances": 13, "executor_memory_gb": 28}
    ok = _hist(best, 1486000, int(0.27 * 104 * 1486000), int(50 * GB))
    x2, x4 = consolidate(ok[0], 2.0), consolidate(ok[0], 4.0)   # ja vistos -> vai ao x0.5
    failed = {**CFG, "executor_cores": 4, "executor_instances": 20, "executor_memory_gb": 16}
    hist = [ok, (x2, ok[1]), (x4, ok[1]), _fail(failed)]
    assert consolidate(ok[0], 0.5).executor_instances == 20      # o x0.5 cairia no formato que falhou
    hy = HybridAdvisor(LLM(CFG), Guardian(CAPS), llm_every=99)
    c = hy.advise(None, "c", None, [], history=hist).config
    assert not (c.executor_cores == 4 and c.executor_instances == 20 and c.executor_memory_gb <= 16)
    # mais memoria no mesmo formato continua permitido
    more = ConfigVector(**{**failed, "executor_memory_gb": 24})
    assert HybridAdvisor._dominated_by_failure(more, [(4, 20, 16)]) is False
    assert HybridAdvisor._dominated_by_failure(ConfigVector(**failed), [(4, 20, 16)]) is True
