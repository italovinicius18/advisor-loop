from advisor_loop.advisor.hybrid import HybridAdvisor, AXES
from advisor_loop.episode import ConfigVector, Episode, Outcome, QueryResult, Recommendation
from advisor_loop.guardian import Guardian

CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}
CFG = dict(executor_cores=8, executor_memory_gb=32, executor_instances=8, driver_memory_gb=8,
           shuffle_partitions=200, advisory_partition_mb=64, aqe_enabled=True,
           broadcast_threshold_mb=10)


class FakeLLM:
    def __init__(self):
        self.calls = 0
    def advise(self, ep, ctx, probe, mems, history=None):
        self.calls += 1
        return Recommendation(rationale="llm", config=ConfigVector(**CFG))


def _ep():
    return Episode(workload="w", dataset="sf100", queries=["q03"], provider="m")


def _out(dur, cpu=None):
    m = {"executor_cpu_time_ms": cpu} if cpu is not None else {}
    return Outcome(status="completed", per_query={"q03": QueryResult(
        status="success", duration_ms=dur, metrics=m)})


def _shape(c):
    return {a: getattr(c, a) for a in AXES}


def test_llm_at_zero_and_every_k_numeric_between():
    llm = FakeLLM()
    hy = HybridAdvisor(llm, Guardian(CAPS), llm_every=3)
    hist = []
    r0 = hy.advise(_ep(), "c", None, [], history=hist)      # n=0 -> LLM
    assert llm.calls == 1
    hist.append((r0.config, _out(60000)))
    r1 = hy.advise(_ep(), "c", None, [], history=hist)      # n=1 -> numerico
    r1_hash = r1.config.config_hash()
    hist.append((r1.config, _out(70000)))
    r2 = hy.advise(_ep(), "c", None, [], history=hist)      # n=2 -> numerico
    assert llm.calls == 1
    assert "pattern-search" in r1.rationale and "pattern-search" in r2.rationale
    hist.append((r2.config, _out(65000)))
    hy.advise(_ep(), "c", None, [], history=hist)           # n=3 -> LLM re-ancora
    assert llm.calls == 2
    assert r1_hash != r2.config.config_hash()               # nunca repete


def test_numeric_step_moves_exactly_one_axis_of_best_and_is_feasible():
    g = Guardian(CAPS)
    hy = HybridAdvisor(FakeLLM(), g)
    best = ConfigVector(**CFG)
    worse = ConfigVector(**{**CFG, "executor_instances": 4})
    hist = [(worse, _out(90000)), (best, _out(60000))]
    r = hy.advise(_ep(), "c", None, [], history=hist)
    diff = [a for a in AXES if getattr(r.config, a) != getattr(best, a)]
    assert len(diff) == 1
    assert r.config.shuffle_partitions == best.shuffle_partitions   # demais knobs preservados
    assert g.validate(r.config)[0]


def test_low_utilization_shrinks_first_high_grows_first():
    hy = HybridAdvisor(FakeLLM(), Guardian(CAPS))
    best = ConfigVector(**CFG)                     # 64 cores totais
    # util = cpu/(64 x 60s): 5% -> encolhe instancias; 90% -> cresce
    low = [(best, _out(60000, cpu=int(0.05 * 64 * 60000)))]
    high = [(best, _out(60000, cpu=int(0.90 * 64 * 60000)))]
    r_low = hy.advise(_ep(), "c", None, [], history=low)
    r_high = hy.advise(_ep(), "c", None, [], history=high)
    assert r_low.config.executor_instances < best.executor_instances
    assert r_high.config.executor_instances > best.executor_instances
    assert "util=5%" in r_low.rationale


def test_falls_back_to_llm_when_no_success_or_no_candidates():
    llm = FakeLLM()
    hy = HybridAdvisor(llm, Guardian(CAPS))
    # so falhas no historico -> LLM
    hist = [(ConfigVector(**CFG), Outcome(status="oom"))]
    hy.advise(_ep(), "c", None, [], history=hist)
    assert llm.calls == 1
    # numeric_step sem candidato inedito -> None (guardian que rejeita tudo)
    class NoGuardian:
        def validate(self, c): return (False, "no")
    hy2 = HybridAdvisor(llm, NoGuardian())
    succ = [(ConfigVector(**CFG), QueryResult(status="success", duration_ms=1000))]
    assert hy2.numeric_step([(ConfigVector(**CFG), _out(1000))], succ) is None
    hy2.advise(_ep(), "c", None, [], history=[(ConfigVector(**CFG), _out(1000))])
    assert llm.calls == 2


def test_step_shrinks_when_bigger_steps_exhausted():
    """Com +-50% ja vistos em todos os eixos, tenta +-25%."""
    g = Guardian(CAPS)
    hy = HybridAdvisor(FakeLLM(), g)
    best = ConfigVector(**CFG)
    hist = [(best, _out(60000))]
    for _ in range(6):                                   # 3 eixos x 2 direcoes a 50%
        r = hy.advise(_ep(), "c", None, [], history=hist)
        assert "passo 50%" in r.rationale
        hist.append((r.config, _out(70000)))             # nada melhora; best fixo
        if len(hist) % 3 == 0:                           # pula as iteracoes do LLM
            hist.append((ConfigVector(**{**CFG, "driver_memory_gb": 4 + len(hist)}), _out(80000)))
    r = hy.numeric_step(hist, [(best, hist[0][1].per_query["q03"])])
    assert "passo 25%" in r.rationale
