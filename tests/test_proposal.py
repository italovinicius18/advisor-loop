from advisor_loop.backends.proposal import ProposalBackend
from advisor_loop.episode import Episode, Recommendation, ConfigVector


def _rec():
    return Recommendation(rationale="proposta inicial", config=ConfigVector(
        executor_cores=4, executor_memory_gb=14, executor_instances=4,
        driver_memory_gb=8, shuffle_partitions=400, advisory_partition_mb=64,
        aqe_enabled=True, broadcast_threshold_mb=10))


def test_execute_returns_proposed_without_lookup():
    """Proposal mode does not grade against an oracle - it just records the proposal."""
    backend = ProposalBackend()
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    out = backend.execute(ep, _rec())
    assert out.status == "proposed"
    assert out.per_query == {}
    assert out.wall_clock_s == 0.0
    assert out.cost_usd == 0.0
