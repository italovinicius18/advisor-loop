from advisor_loop.episode import (
    ConfigVector, task_key, QueryResult, Outcome, Recommendation, Decision, Episode
)
import pydantic, pytest

def test_config_vector_bounds():
    cv = ConfigVector(executor_cores=4, executor_memory_gb=14, executor_instances=4,
                      driver_memory_gb=8, shuffle_partitions=400,
                      advisory_partition_mb=64, aqe_enabled=True, broadcast_threshold_mb=10)
    assert cv.executor_cores == 4
    with pytest.raises(pydantic.ValidationError):
        ConfigVector(executor_cores=99, executor_memory_gb=14, executor_instances=4,
                     driver_memory_gb=8, shuffle_partitions=400,
                     advisory_partition_mb=64, aqe_enabled=True, broadcast_threshold_mb=10)

def test_config_hash_stable_and_task_key():
    kw = dict(executor_cores=4, executor_memory_gb=14, executor_instances=4,
              driver_memory_gb=8, shuffle_partitions=400, advisory_partition_mb=64,
              aqe_enabled=True, broadcast_threshold_mb=10)
    assert ConfigVector(**kw).config_hash() == ConfigVector(**kw).config_hash()
    k = task_key("w", "d100", "q17", ConfigVector(**kw).config_hash(), 2)
    assert len(k) == 16

def test_models_construct_and_validate():
    # QueryResult: valid construction
    qr = QueryResult(status="success", duration_ms=123.45, error=None)
    assert qr.status == "success"
    assert qr.duration_ms == 123.45
    assert qr.error is None

    # QueryResult: invalid status rejected
    with pytest.raises(pydantic.ValidationError):
        QueryResult(status="invalid_status", duration_ms=100.0)

    # Outcome with per_query dict containing QueryResult
    outcome = Outcome(
        status="completed",
        per_query={"q01": QueryResult(status="success", duration_ms=50.0)},
        wall_clock_s=100.5,
        cost_usd=12.34
    )
    assert outcome.status == "completed"
    assert outcome.per_query["q01"].status == "success"
    assert outcome.wall_clock_s == 100.5
    assert outcome.cost_usd == 12.34

    # Outcome defaults
    outcome_default = Outcome(status="completed")
    assert outcome_default.per_query == {}
    assert outcome_default.wall_clock_s == 0.0
    assert outcome_default.cost_usd == 0.0

    # Recommendation with ConfigVector
    cv = ConfigVector(
        executor_cores=8, executor_memory_gb=28, executor_instances=4,
        driver_memory_gb=8, shuffle_partitions=400, advisory_partition_mb=64,
        aqe_enabled=True, broadcast_threshold_mb=10
    )
    rec = Recommendation(
        config=cv,
        rationale="Balanced config for medium workload",
        samples=5
    )
    assert rec.config.executor_cores == 8
    assert rec.rationale == "Balanced config for medium workload"
    assert rec.samples == 5

    # Decision with confidence validation
    dec = Decision(
        action="continue",
        reason="Performance satisfactory",
        evidence=["latency_ok", "throughput_ok"],
        confidence=0.95
    )
    assert dec.action == "continue"
    assert dec.confidence == 0.95
    assert len(dec.evidence) == 2

    # Decision: confidence out of bounds rejected
    with pytest.raises(pydantic.ValidationError):
        Decision(
            action="continue",
            reason="Test",
            evidence=["test"],
            confidence=1.5  # Invalid: > 1.0
        )

    # Episode with defaults
    ep = Episode(
        workload="tpc-ds",
        dataset="sf100",
        queries=["q01", "q02", "q03"],
        provider="gpt-4"
    )
    assert ep.workload == "tpc-ds"
    assert ep.dataset == "sf100"
    assert len(ep.queries) == 3
    assert ep.objective == "min_time"  # default
    assert ep.split == "none"  # default
    assert ep.level == "L0"  # default
    assert ep.provider == "gpt-4"

    # Episode with custom objective and split
    ep_custom = Episode(
        workload="tpc-h",
        dataset="sf500",
        queries=["q05", "q09"],
        objective="feasibility",
        split="loso-scale:sf500",
        level="L2",
        provider="gpt-3.5"
    )
    assert ep_custom.objective == "feasibility"
    assert ep_custom.split == "loso-scale:sf500"
    assert ep_custom.level == "L2"


def test_episode_key_includes_provider_and_level():
    """Two campaigns identical except provider (or level) must NOT collide on resume."""
    from advisor_loop.episode import episode_key
    k_deepseek = episode_key("w", "sf100", "q1", 1, "deepseek", "L0")
    k_sonnet = episode_key("w", "sf100", "q1", 1, "sonnet", "L0")
    assert k_deepseek != k_sonnet

    k_l1 = episode_key("w", "sf100", "q1", 1, "deepseek", "L1")
    k_l2 = episode_key("w", "sf100", "q1", 1, "deepseek", "L2")
    assert k_l1 != k_l2

    # identical-everything -> same key (still allows correct resume-skip)
    assert episode_key("w", "sf100", "q1", 1, "deepseek", "L0") == \
           episode_key("w", "sf100", "q1", 1, "deepseek", "L0")

def test_queryresult_carries_metrics():
    qr = QueryResult(status="success", duration_ms=1200.0,
                     metrics={"input_bytes": 100, "read_throughput_mbps": 5.0})
    assert qr.metrics["input_bytes"] == 100
    assert QueryResult(status="failed").metrics == {}   # default vazio

def test_outcome_status_oom_allowed():
    Outcome(status="oom")   # não levanta
