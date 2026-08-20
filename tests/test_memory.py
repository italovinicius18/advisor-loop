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
