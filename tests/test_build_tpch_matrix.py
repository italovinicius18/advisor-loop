"""Matriz TPC-H (exp06, JSON por celula) -> parquet no schema do oraculo
(analysis.oracle: scale_factor, query, status, duration_ms, executor_cores,
executor_memory_gb, workers), UMA LINHA POR TRY (igual ao training_data TPC-DS)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("build_tpch_matrix", ROOT / "scripts/build_tpch_matrix.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cell(path, cores, mem, workers, recs):
    path.write_text(json.dumps({
        "configuration": {"scale_factor": 500, "workers": workers, "executor_cores": cores,
                          "executor_memory": mem},
        "detailed_results": recs}))


def test_rows_per_try_and_oracle(tmp_path):
    from analysis.oracle import oracle_table, oracle_table_median
    d = tmp_path / "sf500"
    d.mkdir()
    _cell(d / "benchmark-a.json", "12", "48g", 6, [
        {"query": "q09", "try": 1, "status": "success", "execution_time_ms": 480000},
        {"query": "q09", "try": 2, "status": "success", "execution_time_ms": 477000},
        {"query": "q09", "try": 3, "status": "failed", "execution_time_ms": None}])
    _cell(d / "benchmark-b.json", "8", "28g", 6, [
        {"query": "q09", "try": 1, "status": "success", "execution_time_ms": 500000},
        {"query": "q09", "try": 2, "status": "timeout", "execution_time_ms": 1800000}])
    rows = _mod().load_rows(str(tmp_path), sfs=[500])
    assert len(rows) == 5
    assert {(r["executor_cores"], r["executor_memory_gb"], r["workers"]) for r in rows} == {(12, 48.0, 6), (8, 28.0, 6)}
    out = tmp_path / "m.parquet"
    _mod().write_parquet(rows, str(out))
    assert oracle_table(str(out)) == {(500, "q09"): 477000.0}             # min de tries success
    assert oracle_table_median(str(out)) == {(500, "q09"): 478500.0}      # min das medianas
