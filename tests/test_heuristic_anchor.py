"""A0: anchor heuristico sem LLM, plugavel no HybridAdvisor."""
from advisor_loop.advisor.advisor import PINNED_MATRIX, render_profile
from advisor_loop.advisor.heuristic import (HeuristicAnchorAdvisor, heuristic_anchor,
                                            parse_profile)
from advisor_loop.advisor.hybrid import HybridAdvisor
from advisor_loop.episode import ConfigVector, Episode, Outcome, QueryResult
from advisor_loop.guardian import Guardian
from advisor_loop.loop import DEFAULT_CFG

CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}
GB = 1024 ** 3


def _ctx(cpu_ms, shuffle):
    return "## pack\n...\n\n" + render_profile(
        DEFAULT_CFG, 90000, {"executor_cpu_time_ms": cpu_ms, "shuffle_write_bytes": shuffle,
                             "input_bytes": 5 * 1024 ** 2})


def test_parse_profile_reads_the_same_text_the_llm_gets():
    p = parse_profile(_ctx(1_273_000, int(20.7 * GB)))
    assert p["cpu_core_s"] == 1273 and abs(p["shuffle_bytes"] / GB - 20.7) < 0.1
    assert parse_profile("sem perfil") == {}


def test_light_query_gets_few_small_executors():
    # q03-like: 21 core-s, no shuffle -> 8 cores min, 4c/exec, 2g/core
    cfg = heuristic_anchor({"cpu_core_s": 21, "shuffle_bytes": 0}, PINNED_MATRIX, Guardian(CAPS))
    assert (cfg.executor_cores, cfg.executor_instances, cfg.executor_memory_gb) == (4, 2, 8)
    assert cfg.shuffle_partitions == 200            # pinados preservados


def test_shuffle_heavy_query_starts_consolidated():
    # q50@sf250-like: 1273 core-s, 20.7GB shuffle -> ~42 cores -> 12c x 4w x 48g
    cfg = heuristic_anchor({"cpu_core_s": 1273, "shuffle_bytes": 20.7 * GB}, PINNED_MATRIX,
                           Guardian(CAPS))
    assert cfg.executor_cores == 12 and cfg.executor_memory_gb == 48
    assert 3 <= cfg.executor_instances <= 4


def test_guardian_cap_shrinks_instances():
    small = Guardian({**CAPS, "total_cores": 36})
    cfg = heuristic_anchor({"cpu_core_s": 3000, "shuffle_bytes": 5 * GB}, PINNED_MATRIX, small)
    assert small.validate(cfg)[0] and cfg.executor_cores * cfg.executor_instances <= 36


def test_hybrid_with_heuristic_anchor_never_calls_an_llm():
    hy = HybridAdvisor(HeuristicAnchorAdvisor(Guardian(CAPS), PINNED_MATRIX), Guardian(CAPS))
    ep = Episode(workload="w", dataset="sf100", queries=["q50"], provider="none")
    ctx = _ctx(520_000, int(8.8 * GB))
    r0 = hy.advise(ep, ctx, None, [], history=[])
    assert r0.rationale.startswith("A0 heuristico") and r0.config.executor_cores == 12
    out = Outcome(status="completed", per_query={"q50": QueryResult(
        status="success", duration_ms=160000,
        metrics={"executor_cpu_time_ms": 520_000, "shuffle_write_bytes": int(8.8 * GB)})})
    r1 = hy.advise(ep, ctx, None, [], history=[(r0.config, out)])
    assert "pattern-search" in r1.rationale
