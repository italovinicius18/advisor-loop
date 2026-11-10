"""Matriz TPC-DS (training_data.parquet, uma linha por try) -> referencia do estudo
offline (uma linha por celula), no mesmo formato da referencia TPC-H
(experiments/06-tpch-matrix/load_matrix.py): duration_ms = mediana das tries
success; status = success se alguma try deu certo, senao o status mais comum;
telemetria = mediana das tries success. dataset = 'sf<N>'.

Uso: python scripts/build_tpcds_reference.py [--matrix ...] [--out data/matrix/tpcds_reference.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
TELEMETRY = ["shuffle_read_bytes", "shuffle_write_bytes", "input_bytes", "gc_time_ms",
             "num_stages", "num_tasks", "num_joins", "num_broadcast_joins"]


def build(matrix: str, out: str) -> int:
    con = duckdb.connect()
    con.sql(f"CREATE VIEW m AS SELECT * FROM read_parquet('{matrix}')")
    cols = {r[0] for r in con.execute("DESCRIBE m").fetchall()}
    tele = [c for c in TELEMETRY if c in cols]
    tsel = ", ".join(f"median({c}) FILTER (WHERE status='success') AS {c}" for c in tele)
    con.execute(f"""
      COPY (
        SELECT 'sf' || CAST(scale_factor AS INTEGER) AS dataset, query,
               CAST(executor_cores AS BIGINT) AS executor_cores,
               CAST(executor_memory_gb AS BIGINT) AS executor_memory_gb,
               CAST(workers AS BIGINT) AS executor_instances,
               CASE WHEN count(*) FILTER (WHERE status='success') > 0 THEN 'success'
                    ELSE mode(status) END AS status,
               median(duration_ms) FILTER (WHERE status='success') AS duration_ms
               {", " + tsel if tsel else ""}
        FROM m
        GROUP BY ALL
        ORDER BY dataset, query, executor_cores, executor_memory_gb, executor_instances
      ) TO '{out}' (FORMAT parquet)""")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    con.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(ROOT / "data/matrix/training_data.parquet"))
    ap.add_argument("--out", default=str(ROOT / "data/matrix/tpcds_reference.parquet"))
    a = ap.parse_args()
    print("celulas:", build(a.matrix, a.out), "->", a.out)


if __name__ == "__main__":
    main()
