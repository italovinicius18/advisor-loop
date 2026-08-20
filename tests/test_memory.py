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


def test_loso_scale_split_excludes_same_dataset_disjoint_query(tmp_path):
    """split='loso-scale:sfN' is a within-fold leak risk: exclude ALL memories with the
    same dataset regardless of query overlap (default behavior would retain this)."""
    ms = MemoryStore(tmp_path / "mem.jsonl")
    out = Outcome(status="completed", per_query={"q": QueryResult(status="success", duration_ms=1000)})

    # Same dataset, disjoint query -> retained under default split, must be EXCLUDED under loso-scale
    ep_same_ds = Episode(workload="w", dataset="sf100", queries=["q9"], provider="m")
    ms.distill(ep_same_ds, _rec(), out)

    # Different dataset -> retained either way
    ep_other_ds = Episode(workload="w", dataset="sf250", queries=["q9"], provider="m")
    ms.distill(ep_other_ds, _rec(), out)

    retrieve_ep = Episode(workload="w", dataset="sf100", queries=["q2"],
                           split="loso-scale:sf100", provider="m")
    hits = ms.retrieve(retrieve_ep, k=10)
    lessons = "|".join(hits)
    assert "q9" not in lessons or "sf250" in lessons  # sanity: q9 memory only via sf250 line
    assert "sf100" not in lessons, "loso-scale split must exclude ALL same-dataset memories"
    assert "sf250" in lessons, "different-dataset memory must be retained"


def test_loso_query_split_excludes_same_query_different_dataset(tmp_path):
    """split='loso-query:qX' must exclude every memory whose queries intersect ep.queries,
    regardless of dataset (default behavior would retain a different-dataset match)."""
    ms = MemoryStore(tmp_path / "mem.jsonl")
    out = Outcome(status="completed", per_query={"q": QueryResult(status="success", duration_ms=1000)})

    # Different dataset, same query -> retained under default split, must be EXCLUDED under loso-query
    ep_same_q = Episode(workload="w", dataset="sf250", queries=["q2"], provider="m")
    ms.distill(ep_same_q, _rec(), out)

    # Different query -> retained either way
    ep_other_q = Episode(workload="w", dataset="sf250", queries=["q9"], provider="m")
    ms.distill(ep_other_q, _rec(), out)

    retrieve_ep = Episode(workload="w", dataset="sf100", queries=["q2"],
                           split="loso-query:q2", provider="m")
    hits = ms.retrieve(retrieve_ep, k=10)
    lessons = "|".join(hits)
    assert "q2" not in lessons, "loso-query split must exclude memories with overlapping queries"
    assert "q9" in lessons, "non-overlapping-query memory must be retained"


def test_default_split_none_still_uses_pair_semantics(tmp_path):
    """Regression guard: split='none' (default) must keep existing AND/set-intersection
    behavior unchanged."""
    ms = MemoryStore(tmp_path / "mem.jsonl")
    out = Outcome(status="completed", per_query={"q": QueryResult(status="success", duration_ms=1000)})
    ep_a = Episode(workload="w", dataset="sf100", queries=["q1", "q2"], provider="m")
    ms.distill(ep_a, _rec(), out)
    ep_b = Episode(workload="w", dataset="sf100", queries=["q9"], provider="m")
    ms.distill(ep_b, _rec(), out)
    retrieve_ep = Episode(workload="w", dataset="sf100", queries=["q2", "q3"],
                           split="none", provider="m")
    hits = ms.retrieve(retrieve_ep, k=10)
    lessons = "|".join(hits)
    assert "q1,q2" not in lessons
    assert "q9" in lessons
