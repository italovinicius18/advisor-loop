"""Estudo offline em TPC-DS: o prompt nomeia o benchmark certo e a referencia
TPC-DS colapsa tries por celula como a do TPC-H (mediana das tries success)."""
import importlib.util
from pathlib import Path

import duckdb

from advisor_loop.episode import Episode
from advisor_loop.studies.regret import _StudyPack

ROOT = Path(__file__).resolve().parents[1]


def test_study_pack_names_the_benchmark():
    ep = Episode(workload="w", dataset="sf500", queries=["q23"], provider="m")
    assert "TPC-H" in _StudyPack().render(ep)                       # default inalterado
    txt = _StudyPack(benchmark="TPC-DS").render(ep)
    assert "TPC-DS" in txt and "TPC-H" not in txt


def _mod():
    spec = importlib.util.spec_from_file_location("b", ROOT / "scripts/build_tpcds_reference.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_reference_collapses_tries_per_cell(tmp_path):
    src = tmp_path / "m.parquet"
    duckdb.sql("""COPY (SELECT * FROM (VALUES
        (100,'q03',4,14.0,12,'success',20000.0, 1.0e9),
        (100,'q03',4,14.0,12,'success',22000.0, 1.1e9),
        (100,'q03',4,14.0,12,'success',21000.0, 1.2e9),
        (100,'q03',2,7.0,2,'timeout',1800000.0, NULL),
        (100,'q03',2,7.0,2,'timeout',1800000.0, NULL),
        (500,'q03',12,48.0,4,'failed',NULL, NULL)
      ) t(scale_factor, query, executor_cores, executor_memory_gb, workers, status, duration_ms,
          shuffle_read_bytes)) TO '""" + str(src) + "' (FORMAT parquet)")
    out = tmp_path / "ref.parquet"
    _mod().build(str(src), str(out))
    rows = duckdb.sql(f"""SELECT dataset, query, executor_cores, executor_memory_gb, executor_instances,
                          status, duration_ms, shuffle_read_bytes FROM read_parquet('{out}')
                          ORDER BY dataset, executor_cores""").fetchall()
    assert rows == [
        ("sf100", "q03", 2, 7, 2, "timeout", None, None),
        ("sf100", "q03", 4, 14, 12, "success", 21000.0, 1.1e9),        # mediana das tries
        ("sf500", "q03", 12, 48, 4, "failed", None, None),
    ]
