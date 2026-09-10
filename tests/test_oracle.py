import pandas as pd
from analysis.oracle import oracle_table, best_config


def test_oracle_min_success(tmp_path):
    df = pd.DataFrame([
        {"scale_factor": 100, "query": "q23", "duration_ms": 300.0, "status": "success"},
        {"scale_factor": 100, "query": "q23", "duration_ms": 120.0, "status": "success"},
        {"scale_factor": 100, "query": "q23", "duration_ms": 90.0, "status": "failed"},  # ignored
        {"scale_factor": 100, "query": "q50", "duration_ms": 500.0, "status": "success"},
    ])
    p = tmp_path / "m.parquet"
    df.to_parquet(p)
    t = oracle_table(str(p))
    assert t[(100, "q23")] == 120.0
    assert t[(100, "q50")] == 500.0
    assert (100, "q23") in t and (100, "q50") in t


def test_oracle_median_is_min_of_per_cell_medians(tmp_path):
    from analysis.oracle import oracle_table_median
    cell_a = {"scale_factor": 100, "query": "q23", "status": "success",
              "executor_cores": 4, "executor_memory_gb": 14, "workers": 6}
    cell_b = {**cell_a, "executor_cores": 8}
    df = pd.DataFrame([
        {**cell_a, "duration_ms": 100.0},   # celula A: 100, 300, 320 -> mediana 300
        {**cell_a, "duration_ms": 300.0},
        {**cell_a, "duration_ms": 320.0},
        {**cell_b, "duration_ms": 200.0},   # celula B: 200, 210, 220 -> mediana 210
        {**cell_b, "duration_ms": 210.0},
        {**cell_b, "duration_ms": 220.0},
    ])
    p = tmp_path / "m.parquet"
    df.to_parquet(p)
    assert oracle_table(str(p))[(100, "q23")] == 100.0          # min puro: a sortuda de A
    assert oracle_table_median(str(p))[(100, "q23")] == 210.0   # justa: melhor celula pela mediana


def test_best_config_argmin(tmp_path):
    df = pd.DataFrame([
        {"scale_factor": 100, "query": "q23", "duration_ms": 300.0, "status": "success",
         "executor_cores": 4, "executor_memory_gb": 14, "workers": 6},
        {"scale_factor": 100, "query": "q23", "duration_ms": 120.0, "status": "success",
         "executor_cores": 8, "executor_memory_gb": 28, "workers": 8},
    ])
    p = tmp_path / "m.parquet"
    df.to_parquet(p)
    bc = best_config(str(p), 100, "q23")
    assert bc["duration_ms"] == 120.0
    assert bc["executor_cores"] == 8
    assert bc["workers"] == 8
