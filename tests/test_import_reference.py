import duckdb
import json
from pathlib import Path
from scripts.import_reference import convert_dir


def test_convert(tmp_path):
    src = {
        "configuration": {
            "scale_factor": 100,
            "executor_cores": "4",
            "executor_memory": "14g",
            "workers": 4,
        },
        "detailed_results": [
            {
                "query": "q1",
                "try": 1,
                "status": "success",
                "execution_time_ms": 1234.5,
            }
        ],
    }
    (tmp_path / "b.json").write_text(json.dumps(src))
    out = tmp_path / "ref.parquet"
    n = convert_dir(tmp_path, out)
    assert n == 1
    row = duckdb.sql(f"SELECT * FROM read_parquet('{out}')").fetchone()
    assert row == ("sf100", "q1", 4, 14, 4, "success", 1234.5)
