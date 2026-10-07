"""v4 scale-aware: medicao Delta, envelope, baseline dimensionada, random e
loop dentro do envelope, timeout relativo. Validacao contra a matriz (as
menores celulas que completaram por SF devem passar no piso do Guardian; a
baseline default 8g deve ser rejeitada a partir de sf500)."""
import io
import json
import pathlib
import duckdb
import pytest
from advisor_loop.advisor.advisor import PINNED_MATRIX
from advisor_loop.episode import ConfigVector, Episode, Outcome, QueryResult, Recommendation
from advisor_loop.guardian import Guardian
from advisor_loop.loop import DEFAULT_CFG, optimize
from advisor_loop.sizing import (Envelope, Sizer, delta_snapshot_bytes, make_envelope,
                                 query_data_gb, tables_in_sql)
from advisor_loop.studies.advisors import FreeRandomAdvisor
from tests.test_loop_tries import _Deps

CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}
MATRIX = pathlib.Path(__file__).resolve().parents[1] / "data/matrix/training_data.parquet"
MEASURED_GB = {  # snapshot Delta atual das tabelas de cada query (MinIO, 2026-09-24)
    (10, "q03"): 1.7, (10, "q23"): 3.6, (10, "q50"): 1.9,
    (50, "q03"): 8.3, (50, "q23"): 17.9, (50, "q50"): 9.2,
    (100, "q03"): 16.8, (100, "q23"): 36.6, (100, "q50"): 18.7,
    (250, "q03"): 41.5, (250, "q23"): 89.6, (250, "q50"): 46.5,
    (500, "q03"): 78.5, (500, "q23"): 173.1, (500, "q50"): 88.0,
    (750, "q03"): 118.3, (750, "q23"): 261.0, (750, "q50"): 132.8,
    (1000, "q03"): 169.0, (1000, "q23"): 369.1, (1000, "q50"): 189.6,
}
# onde o default 8c x 1w x 8g NAO completou no nosso harness (200 min) -> deve ser rejeitado
DEFAULT_FAILED = {(500, "q23"), (500, "q50")} | {(sf, q) for sf in (750, 1000) for q in ("q03", "q23", "q50")}
DEFAULT_BORDERLINE = {(250, "q23")}   # completou em 27 min; rejeitar e aceitavel


class FakeS3:
    """list_objects_v2/get_object minimos sobre um dict 'bucket/key'->bytes."""
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        full = f"{Bucket}/{Prefix}"
        return {"Contents": [{"Key": k[len(Bucket) + 1:], "Size": len(v)}
                             for k, v in self.objects.items() if k.startswith(full)]}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[f"{Bucket}/{Key}"])}


def _commit(adds: dict[str, int], removes=()):
    lines = [json.dumps({"commitInfo": {"operation": "WRITE"}})]
    lines += [json.dumps({"remove": {"path": p}}) for p in removes]
    lines += [json.dumps({"add": {"path": p, "size": s}}) for p, s in adds.items()]
    return ("\n".join(lines) + "\n").encode()


def test_tables_in_sql_word_boundaries():
    sql = "select * from store_sales ss join date_dim d on ... where store_returns_x = 1"
    known = ["store", "store_sales", "date_dim", "store_returns", "item"]
    assert tables_in_sql(sql, known) == ["store_sales", "date_dim"]


def test_delta_snapshot_applies_adds_and_removes_in_order():
    s3 = FakeS3({
        "silver/sf1/delta/parquet/t/_delta_log/00000000000000000000.json": _commit({"a.parquet": 100, "b.parquet": 200}),
        "silver/sf1/delta/parquet/t/_delta_log/00000000000000000001.json": _commit({"c.parquet": 50}, removes=["a.parquet", "b.parquet"]),
        "silver/sf1/delta/parquet/t/part-a.parquet": b"x" * 100,
    })
    assert delta_snapshot_bytes(s3, "silver", "sf1/delta/parquet/t") == 50   # so o snapshot atual
    gb, per = query_data_gb(s3, {"t": "s3://silver/sf1/delta/parquet/t"}, "select 1 from t")
    assert per["t"] * 1024 ** 3 == 50 and gb == 0.0


def test_delta_snapshot_with_checkpoint_falls_back_to_listing():
    s3 = FakeS3({
        "b/p/t/_delta_log/_last_checkpoint": b"{}",
        "b/p/t/_delta_log/00000000000000000010.checkpoint.parquet": b"z" * 10,
        "b/p/t/part-1.parquet": b"x" * 300, "b/p/t/part-2.parquet": b"x" * 700,
    })
    gb, per = query_data_gb(s3, {"t": "s3://b/p/t"}, "from t")
    assert per["t"] * 1024 ** 3 == 1000


def test_envelope_floors_admit_matrix_minima_and_reject_spark_default_where_it_failed():
    con = duckdb.connect()
    for (sf, q), gb in MEASURED_GB.items():
        env = make_envelope(gb)
        row = con.execute(
            "SELECT executor_cores, CAST(executor_memory_gb AS INT), workers, workers*executor_memory_gb AS m "
            "FROM read_parquet(?) WHERE status='success' AND CAST(scale_factor AS INT)=? AND query=? "
            "ORDER BY m LIMIT 1", [str(MATRIX), sf, q]).fetchone()
        c, m, w, _ = row
        cell = ConfigVector(executor_cores=int(c), executor_memory_gb=int(m), executor_instances=int(w), **PINNED_MATRIX)
        assert env.check(cell)[0], f"sf{sf} {q}: menor celula que completou ({c}cx{w}wx{m}g) rejeitada pelo piso"
        default_ok = env.check(DEFAULT_CFG)[0]
        if (sf, q) in DEFAULT_FAILED:
            assert not default_ok, f"sf{sf} {q}: default deveria ser rejeitado (estourou 200 min)"
        elif (sf, q) not in DEFAULT_BORDERLINE:
            assert default_ok, f"sf{sf} {q}: default completou e foi rejeitado"


def test_envelope_baseline_is_inside_and_scales():
    for (sf, q), gb in MEASURED_GB.items():
        env = make_envelope(gb)
        b = env.baseline(PINNED_MATRIX, CAPS)
        assert env.check(b)[0] and Guardian(CAPS).validate(b)[0]
        assert b.executor_cores == 4 and b.shuffle_partitions == 200
    small = make_envelope(MEASURED_GB[(10, "q03")]).baseline(PINNED_MATRIX, CAPS)
    big = make_envelope(MEASURED_GB[(1000, "q23")]).baseline(PINNED_MATRIX, CAPS)
    assert small.executor_instances * small.executor_memory_gb < big.executor_instances * big.executor_memory_gb
    # q23@sf1000 (369 GB): 4 GB/core -> 4c x 12w x 16g = 192 GB = menor celula que completou na matriz
    assert (big.executor_memory_gb, big.executor_instances) == (16, 12)


def test_random_samples_inside_envelope():
    env = make_envelope(173)
    adv = FreeRandomAdvisor(Guardian(CAPS), seed=1)
    adv.set_envelope(env)
    ep = Episode(workload="w", dataset="sf500", queries=["q23"], provider="m")
    hist = []
    for _ in range(8):
        rec = adv.advise(ep, "c", None, [], history=hist)
        assert env.check(rec.config)[0]
        hist.append((rec.config, None))


class _Backend:
    in_session_tries = True
    def __init__(self):
        self.calls = []
    def execute(self, ep, rec, tries=1, wall_clock_s=None):
        self.calls.append((rec.config, wall_clock_s))
        return Outcome(status="completed", per_query={ep.queries[0]: QueryResult(
            status="success", duration_ms=600000.0, metrics={"executor_cpu_time_ms": 5_000_000})})


class _Adv:
    def __init__(self):
        self.env, self.ctx = None, []
    def set_envelope(self, env):
        self.env = env
    def advise(self, ep, ctx, probe, mems, history=[]):
        self.ctx.append(ctx)
        # 1a proposta abaixo do piso (8g total) -> deve virar invalid_recommendation
        n = len(history)
        return Recommendation(rationale="", config=ConfigVector(**{**PINNED_MATRIX, "executor_cores": 4,
                                                                    "executor_memory_gb": 8, "executor_instances": 1 if n == 0 else 12}))


def test_loop_uses_envelope_for_context_validation_and_relative_timeout():
    env = make_envelope(173)
    be, adv = _Backend(), _Adv()
    ep = Episode(workload="w", dataset="sf500", queries=["q23"], provider="m")
    r = optimize(ep, _Deps(be), adv, max_iters=2, patience=3, tries=3,
                 baseline_cfg=env.baseline(PINNED_MATRIX, CAPS), envelope=env)
    assert adv.env is env
    assert "Envelope de escala" in adv.ctx[0] and "173" in adv.ctx[0]
    assert r["iterations"][0]["status"] == "invalid_recommendation"
    assert "escala" in r["iterations"][0]["reason"]
    assert r["iterations"][1]["status"] == "completed"
    # teto relativo = 3 x baseline (600s) = 1800s, passado ao backend na proposta
    assert be.calls[1][1] == pytest.approx(1800.0)


def test_sizer_reads_tpch_sql_from_its_own_dir(tmp_path):
    # tpch-sf500 q09 nao pode ler queries/q09.sql (TPC-DS): le queries-tpch/
    (tmp_path / "queries").mkdir()
    (tmp_path / "queries-tpch").mkdir()
    (tmp_path / "queries" / "q09.sql").write_text("select 1 from store_sales")
    (tmp_path / "queries-tpch" / "q09.sql").write_text("select 1 from lineitem")
    s3 = FakeS3({"b/h/lineitem/_delta_log/00000000000000000000.json": _commit({"a": 5 * 1024 ** 3})})
    sz = Sizer(s3, {"tpch-sf500": {"lineitem": "s3://b/h/lineitem"}}, tmp_path / "queries")
    env = sz(Episode(workload="w", dataset="tpch-sf500", queries=["q09"], provider="m"))
    assert isinstance(env, Envelope) and env.data_gb == 5.0


def test_sizer_caches_and_degrades_gracefully(tmp_path):
    (tmp_path / "q03.sql").write_text("select 1 from t")
    s3 = FakeS3({"b/p/t/_delta_log/00000000000000000000.json": _commit({"a": 3 * 1024 ** 3})})
    sz = Sizer(s3, {"sf1": {"t": "s3://b/p/t"}}, tmp_path)
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    env = sz(ep)
    assert isinstance(env, Envelope) and env.data_gb == 3.0 and sz(ep) is env
    assert sz(Episode(workload="w", dataset="sf9", queries=["q03"], provider="m")) is None
