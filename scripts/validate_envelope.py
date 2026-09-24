"""Validacao OFFLINE do envelope v4 contra a matriz (calibracao do harness; a
matriz nao entra no loop). Para cada (SF, query): menor celula que completou,
se passa no piso do Guardian, se a baseline default passa, e a baseline v4.

Uso: python scripts/validate_envelope.py [--data-gb sf=gb ...]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duckdb
from advisor_loop.advisor.advisor import PINNED_MATRIX
from advisor_loop.episode import ConfigVector
from advisor_loop.loop import DEFAULT_CFG
from advisor_loop.sizing import make_envelope

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = {10: 8.7, 50: 42, 100: 84, 250: 200, 500: 420, 750: 336, 1000: 479}
CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(ROOT / "data/matrix/training_data.parquet"))
    ap.add_argument("--data-gb", nargs="*", default=[], help="sf=gb overrides")
    a = ap.parse_args()
    data = dict(DEFAULT_DATA)
    for kv in a.data_gb:
        k, v = kv.split("=")
        data[int(k)] = float(v)
    con = duckdb.connect()
    print(f"{'sf':>5} {'query':5} {'data_gb':>7} {'piso mem/cores':>15} {'menor celula ok':>16} {'passa':>5} "
          f"{'default':>7} {'baseline v4':>16} {'falhas matriz':>13}")
    for sf, gb in sorted(data.items()):
        env = make_envelope(gb)
        base = env.baseline(PINNED_MATRIX, CAPS)
        for q in ("q03", "q23", "q50"):
            row = con.execute(
                "SELECT executor_cores, CAST(executor_memory_gb AS INT), workers, workers*executor_memory_gb AS m "
                "FROM read_parquet(?) WHERE status='success' AND CAST(scale_factor AS INT)=? AND query=? "
                "ORDER BY m LIMIT 1", [a.matrix, sf, q]).fetchone()
            bad = con.execute("SELECT count(DISTINCT executor_cores||'-'||workers||'-'||executor_memory_gb) "
                              "FROM read_parquet(?) WHERE status<>'success' AND CAST(scale_factor AS INT)=? AND query=?",
                              [a.matrix, sf, q]).fetchone()[0]
            if row is None:
                continue
            c, m, w, _ = row
            cell = ConfigVector(executor_cores=int(c), executor_memory_gb=int(m), executor_instances=int(w), **PINNED_MATRIX)
            print(f"{sf:>5} {q:5} {gb:7.1f} {env.floor_mem_gb:6.0f}g/{env.floor_cores:<3}c "
                  f"{f'{c}cx{w}wx{m}g':>16} {'ok' if env.check(cell)[0] else 'REJ':>5} "
                  f"{'ok' if env.check(DEFAULT_CFG)[0] else 'REJ':>7} "
                  f"{f'{base.executor_cores}cx{base.executor_instances}wx{base.executor_memory_gb}g':>16} {bad:>13}")


if __name__ == "__main__":
    main()
