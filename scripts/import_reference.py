import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def convert_dir(src_dir: str | Path, out: str | Path) -> int:
    rows = []
    for f in sorted(Path(src_dir).rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        c = d.get("configuration") or {}
        if not c or "detailed_results" not in d:
            continue
        mem = int(re.sub(r"\D", "", str(c["executor_memory"])))
        for r in d["detailed_results"]:
            rows.append(
                {
                    "dataset": f"sf{c['scale_factor']}",
                    "query": r["query"],
                    "executor_cores": int(c["executor_cores"]),
                    "executor_memory_gb": mem,
                    "executor_instances": int(c["workers"]),
                    "status": r["status"],
                    "duration_ms": r.get("execution_time_ms"),
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), str(out))
    return len(rows)


if __name__ == "__main__":
    print(convert_dir(sys.argv[1], sys.argv[2]))
