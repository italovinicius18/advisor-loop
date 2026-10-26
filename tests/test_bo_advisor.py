"""BO (GP + expected improvement) sobre as celulas da grade, estilo CherryPick."""
import math

from advisor_loop.episode import ConfigVector, Episode, Outcome, QueryResult
from advisor_loop.studies.advisors import BOAdvisor

PIN = dict(driver_memory_gb=8, shuffle_partitions=200, advisory_partition_mb=64,
           aqe_enabled=True, broadcast_threshold_mb=10)
SPACE = [ConfigVector(executor_cores=c, executor_memory_gb=4 * c, executor_instances=w, **PIN)
         for c in (2, 4, 8, 12, 24) for w in (2, 4, 6, 8, 12, 20) if c * w <= 120]
EP = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")


def _time(cv):
    # funcao suave com otimo em (8 cores, 6 workers); forma infeasivel: 24 cores x 4+ workers
    if cv.executor_cores == 24 and cv.executor_instances >= 4:
        return None
    return 100 * (1 + (math.log2(cv.executor_cores) - 3) ** 2 + 0.5 * (math.log2(cv.executor_instances) - math.log2(6)) ** 2)


def _run(adv, budget):
    hist = []
    for _ in range(budget):
        cv = adv.advise(EP, "", None, [], history=hist).config
        t = _time(cv)
        out = (Outcome(status="infeasible", per_query={}) if t is None else
               Outcome(status="completed", per_query={"q1": QueryResult(status="success", duration_ms=t)}))
        hist.append((cv, out))
    return hist


def test_bo_is_deterministic_and_never_repeats():
    h1 = _run(BOAdvisor(SPACE, seed=3), 8)
    h2 = _run(BOAdvisor(SPACE, seed=3), 8)
    assert [c.config_hash() for c, _ in h1] == [c.config_hash() for c, _ in h2]
    assert len({c.config_hash() for c, _ in h1}) == 8


def test_bo_finds_optimum_more_often_than_its_random_start():
    found = 0
    for seed in range(20):
        hist = _run(BOAdvisor(SPACE, seed=seed), 8)
        best = min(o.per_query["q1"].duration_ms for _, o in hist if o.per_query)
        found += best <= 100.0 + 1e-9
    assert found >= 14      # random com 8 de 27 celulas acharia o otimo em ~30% das vezes


def test_bo_survives_all_infeasible_start():
    adv = BOAdvisor(SPACE, seed=0)
    bad = [c for c in SPACE if _time(c) is None]
    hist = [(c, Outcome(status="infeasible", per_query={})) for c in bad[:2]]
    cv = adv.advise(EP, "", None, [], history=hist).config
    assert cv.config_hash() not in {c.config_hash() for c, _ in hist}
