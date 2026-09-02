"""ReplayBackend expõe telemetria do parquet em QueryResult.metrics (ablação
A3t vs A3m). Parquet sem colunas de telemetria => metrics vazio (compat)."""
import duckdb
from advisor_loop.backends.replay import ReplayBackend
from advisor_loop.episode import Episode, Recommendation, ConfigVector


def _rec(cores, mem, inst):
    return Recommendation(rationale="", config=ConfigVector(
        executor_cores=cores, executor_memory_gb=mem, executor_instances=inst,
        driver_memory_gb=8, shuffle_partitions=400, advisory_partition_mb=64,
        aqe_enabled=True, broadcast_threshold_mb=10))


def _ref_with_metrics(tmp_path):
    p = tmp_path / "ref.parquet"
    duckdb.sql("""SELECT * FROM (VALUES
        ('sf100','q1',4,14,4,'success',1200.0, 100, 200, 999, 42))
        t(dataset,query,executor_cores,executor_memory_gb,executor_instances,
          status,duration_ms,shuffle_read_bytes,shuffle_write_bytes,input_bytes,gc_time_ms)
    """).write_parquet(str(p))
    return p


def test_metrics_surfaced_when_columns_present(tmp_path):
    rb = ReplayBackend(_ref_with_metrics(tmp_path))
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    out = rb.execute(ep, _rec(4, 14, 4))
    qr = out.per_query["q1"]
    assert qr.duration_ms == 1200.0
    assert qr.metrics["shuffle_read_bytes"] == 100
    assert qr.metrics["gc_time_ms"] == 42
    assert "num_joins" not in qr.metrics       # coluna ausente => não aparece


def test_emit_metrics_false_suppresses_telemetry(tmp_path):
    # A3t: mesmo com telemetria no parquet, metrics fica vazio.
    rb = ReplayBackend(_ref_with_metrics(tmp_path), emit_metrics=False)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    out = rb.execute(ep, _rec(4, 14, 4))
    assert out.per_query["q1"].duration_ms == 1200.0
    assert out.per_query["q1"].metrics == {}


def test_metrics_empty_when_no_telemetry_columns(tmp_path):
    p = tmp_path / "bare.parquet"
    duckdb.sql("""SELECT * FROM (VALUES ('sf100','q1',4,14,4,'success',900.0))
        t(dataset,query,executor_cores,executor_memory_gb,executor_instances,status,duration_ms)
    """).write_parquet(str(p))
    rb = ReplayBackend(p)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    out = rb.execute(ep, _rec(4, 14, 4))
    assert out.per_query["q1"].metrics == {}    # compat: sem telemetria
