"""Matriz TPC-H (exp06) -> data/matrix/tpch_training_data.parquet (regua do TPC-H online).

Uso: python scripts/build_tpch_matrix.py --metrics <dir com sf500/, sf750/> --sfs 500 750
Le os JSON por celula (`configuration` + `detailed_results`, 3 tries x 22 queries)
puxados do MinIO (spark-logs/benchmarks/tpch/) e grava UMA LINHA POR TRY no
schema que analysis.oracle e scripts/gen_sf_campaigns.py ja leem. Mesma regua do
TPC-DS: t* = min das tries success; t*_median = min das medianas por celula.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mem_gb(v) -> float:
    return float(str(v).lower().removesuffix("g"))


def load_rows(metrics_dir: str, sfs: list[int]) -> list[dict]:
    rows = []
    for sf in sfs:
        for f in sorted(glob.glob(os.path.join(metrics_dir, f"sf{sf}", "benchmark-*.json"))):
            d = json.load(open(f))
            c = d["configuration"]
            for r in d.get("detailed_results", []):
                rows.append({
                    "scale_factor": int(c.get("scale_factor", sf)),
                    "query": r["query"],
                    "try": r.get("try"),
                    "status": r.get("status"),
                    "duration_ms": (float(r["execution_time_ms"])
                                    if r.get("execution_time_ms") is not None else None),
                    "executor_cores": int(c["executor_cores"]),
                    "executor_memory_gb": _mem_gb(c["executor_memory"]),
                    "workers": int(c["workers"]),
                    "source_file": os.path.basename(f),
                })
    return rows


def write_parquet(rows: list[dict], path: str) -> None:
    import duckdb
    con = duckdb.connect()
    con.execute("""CREATE TABLE m (scale_factor BIGINT, query VARCHAR, try BIGINT, status VARCHAR,
                   duration_ms DOUBLE, executor_cores BIGINT, executor_memory_gb DOUBLE,
                   workers BIGINT, source_file VARCHAR)""")
    con.executemany("INSERT INTO m VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [[r[k] for k in ("scale_factor", "query", "try", "status", "duration_ms",
                                     "executor_cores", "executor_memory_gb", "workers",
                                     "source_file")] for r in rows])
    con.execute("COPY m TO ? (FORMAT parquet)", [path])
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--sfs", nargs="+", type=int, default=[500, 750])
    ap.add_argument("--out", default=str(ROOT / "data/matrix/tpch_training_data.parquet"))
    a = ap.parse_args()
    rows = load_rows(a.metrics, a.sfs)
    write_parquet(rows, a.out)
    print(f"wrote {len(rows)} linhas -> {a.out}")


if __name__ == "__main__":
    main()
