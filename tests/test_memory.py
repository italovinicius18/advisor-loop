from advisor_loop.memory.store import MemoryStore
from advisor_loop.episode import Episode, Outcome, QueryResult, Recommendation, ConfigVector

def _ep(dataset, query):
    return Episode(workload="w", dataset=dataset, queries=[query], provider="m")

def _rec():
    return Recommendation(rationale="", config=ConfigVector(
        executor_cores=4, executor_memory_gb=14, executor_instances=4, driver_memory_gb=8,
        shuffle_partitions=400, advisory_partition_mb=64, aqe_enabled=True,
        broadcast_threshold_mb=10))

def test_distill_and_leakage_filter(tmp_path):
    ms = MemoryStore(tmp_path / "mem.jsonl")
    out = Outcome(status="completed", per_query={"q1": QueryResult(status="success", duration_ms=1000)})
    ms.distill(_ep("sf100", "q1"), _rec(), out)
    ms.distill(_ep("sf250", "q2"), _rec(), out)
    hits = ms.retrieve(_ep("sf100", "q1"))
    assert len(hits) == 1 and "sf250" in hits[0]      # memoria do proprio (sf100,q1) excluida

def test_leakage_filter_and_semantics(tmp_path):
    """Validate AND/set-intersection semantics: exclude ONLY when BOTH dataset AND query overlap."""
    ms = MemoryStore(tmp_path / "mem.jsonl")
    out = Outcome(status="completed", per_query={"q": QueryResult(status="success", duration_ms=1000)})

    # Memory A: batch with multiple queries
    ep_a = Episode(workload="w", dataset="sf100", queries=["q1", "q2"], provider="m")
    ms.distill(ep_a, _rec(), out)

    # Memory B: same dataset, disjoint queries
    ep_b = Episode(workload="w", dataset="sf100", queries=["q9"], provider="m")
    ms.distill(ep_b, _rec(), out)

    # Memory C: different dataset, same query
    ep_c = Episode(workload="w", dataset="sf250", queries=["q2"], provider="m")
    ms.distill(ep_c, _rec(), out)

    # Retrieve: sf100 with queries [q2, q3]
    retrieve_ep = Episode(workload="w", dataset="sf100", queries=["q2", "q3"], provider="m")
    hits = ms.retrieve(retrieve_ep, k=10)

    # A should be EXCLUDED: same dataset + overlap via q2
    # B should be RETAINED: same dataset but NO overlap (q9 ∉ {q2,q3})
    # C should be RETAINED: different dataset (no leakage even though q2 matches)
    lessons = "|".join(hits)
    assert "q1,q2" not in lessons, "Memory A (sf100,[q1,q2]) must be excluded due to dataset+query overlap"
    assert "q9" in lessons, "Memory B (sf100,[q9]) must be retained: same dataset but disjoint queries"
    assert "sf250" in lessons, "Memory C (sf250,[q2]) must be retained: different dataset is not leakage"
