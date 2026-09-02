"""Throwaway Phase-2 pilot worker (spike, not kept infra).

Runs ONE TPC-H query on a fresh local[*] SparkSession under a proposed
ConfigVector and prints JSON: {status, duration_ms, metrics, plan}.

A fresh process per run is deliberate: driver memory and local[N] can't
change after the JVM boots, and one query per app means every Spark stage
belongs to that query, so summing the REST metrics attributes cleanly.

Real telemetry comes from the live Spark UI REST API (/api/v1) before the
session stops; the physical plan is the AQE-executed plan, distilled to the
lines that map to the tunable knobs (Exchange / joins / scans / aggregates).
"""
import json
import sys
import time
import urllib.request

from pyspark.sql import SparkSession

CORE_CAP = 8      # leave headroom on the 28-core / 15 GB WSL box
MEM_CAP_GB = 6    # ~7 GB free; keep the JVM heap under it

TABLES = ["customer", "lineitem", "nation", "orders", "part",
          "partsupp", "region", "supplier"]


def _fetch(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _collect_metrics(spark, plan_text):
    """Sum real stage/executor metrics from the live Spark UI REST API."""
    try:
        base = spark.sparkContext.uiWebUrl
        app_id = spark.sparkContext.applicationId
        stages = _fetch(f"{base}/api/v1/applications/{app_id}/stages")
        done = [s for s in stages if s.get("status") == "COMPLETE"]
        execs = _fetch(f"{base}/api/v1/applications/{app_id}/executors")
        m = {
            "shuffle_read_bytes": sum(s.get("shuffleReadBytes", 0) for s in done),
            "shuffle_write_bytes": sum(s.get("shuffleWriteBytes", 0) for s in done),
            "input_bytes": sum(s.get("inputBytes", 0) for s in done),
            "memory_spilled_bytes": sum(s.get("memoryBytesSpilled", 0) for s in done),
            "disk_spilled_bytes": sum(s.get("diskBytesSpilled", 0) for s in done),
            "num_stages": len(done),
            "num_tasks": sum(s.get("numCompleteTasks", 0) for s in done),
            "gc_time_ms": sum(e.get("totalGCTime", 0) for e in execs),
        }
    except Exception as e:  # metrics are best-effort; timing is the hard signal
        m = {"metrics_error": str(e)}
    # join shape from the plan text — what broadcast_threshold / AQE act on
    m["num_joins"] = plan_text.count("Join")
    m["num_broadcast_joins"] = plan_text.count("BroadcastHashJoin")
    m["num_exchanges"] = plan_text.count("Exchange")
    return m


def _distill_plan(plan_text, max_lines=45):
    """Keep only the plan lines that map to a knob; drop project/filter noise."""
    keep = ("Exchange", "Join", "Scan", "HashAggregate", "Sort", "Aggregate",
            "AQEShuffleRead", "BroadcastExchange")
    lines = [ln for ln in plan_text.splitlines() if any(k in ln for k in keep)]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... (+{len(lines) - max_lines} more)"]
    return "\n".join(lines)


def main():
    data_dir = sys.argv[1]
    query_sql = sys.argv[2]
    cfg = json.loads(sys.argv[3])
    out_path = sys.argv[4]

    cores = min(int(cfg["executor_cores"]), CORE_CAP)
    mem_gb = min(int(cfg["driver_memory_gb"]), MEM_CAP_GB)

    b = (SparkSession.builder
         .master(f"local[{cores}]")
         .appName("phase2-pilot")
         .config("spark.driver.memory", f"{mem_gb}g")
         .config("spark.sql.shuffle.partitions", int(cfg["shuffle_partitions"]))
         .config("spark.sql.adaptive.enabled", bool(cfg["aqe_enabled"]))
         .config("spark.sql.adaptive.advisoryPartitionSizeInBytes",
                 f"{int(cfg['advisory_partition_mb'])}m")
         .config("spark.ui.showConsoleProgress", "false"))
    bt = int(cfg["broadcast_threshold_mb"])
    b = b.config("spark.sql.autoBroadcastJoinThreshold",
                 -1 if bt < 0 else f"{bt}m")
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    result = {"status": "failed", "duration_ms": None, "metrics": {}, "plan": ""}
    try:
        for t in TABLES:
            spark.read.parquet(f"{data_dir}/{t}.parquet").createOrReplaceTempView(t)
        df = spark.sql(query_sql)
        plan_text = df._jdf.queryExecution().executedPlan().toString()
        t0 = time.time()
        rows = df.collect()  # TPC-H result sets are tiny; forces full execution
        dur_ms = (time.time() - t0) * 1000.0
        # re-read the post-AQE plan (adaptive rewrites it during execution)
        plan_text = df._jdf.queryExecution().executedPlan().toString()
        result = {
            "status": "success",
            "duration_ms": dur_ms,
            "rows": len(rows),
            "metrics": _collect_metrics(spark, plan_text),
            "plan": _distill_plan(plan_text),
        }
    except Exception as e:
        msg = str(e)
        status = "oom" if "OutOfMemory" in msg or "GC overhead" in msg else "failed"
        result = {"status": status, "duration_ms": None, "metrics": {},
                  "plan": "", "error": msg[:800]}
    finally:
        spark.stop()

    with open(out_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
