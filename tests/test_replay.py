import duckdb
from pathlib import Path
from advisor_loop.backends.replay import ReplayBackend
from advisor_loop.episode import Episode, Recommendation, ConfigVector

def _ref(tmp_path) -> Path:
    p = tmp_path / "ref.parquet"
    duckdb.sql("""SELECT * FROM (VALUES
        ('sf100','q1',4,14,4,'success',1200.0),
        ('sf100','q1',8,28,2,'success', 900.0),
        ('sf100','q2',4,14,4,'timeout', NULL))
        t(dataset,query,executor_cores,executor_memory_gb,executor_instances,status,duration_ms)
    """).write_parquet(str(p))
    return p

def _rec(cores, mem, inst):
    return Recommendation(rationale="", config=ConfigVector(
        executor_cores=cores, executor_memory_gb=mem, executor_instances=inst,
        driver_memory_gb=8, shuffle_partitions=400, advisory_partition_mb=64,
        aqe_enabled=True, broadcast_threshold_mb=10))

def test_execute_hit_miss_and_oracle(tmp_path):
    rb = ReplayBackend(_ref(tmp_path))
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    out = rb.execute(ep, _rec(4, 14, 4))
    assert out.per_query["q1"].duration_ms == 1200.0
    assert rb.execute(ep, _rec(12, 48, 6)).status == "infeasible"   # celula nao medida
    assert rb.oracle("sf100", "q1") == 900.0
