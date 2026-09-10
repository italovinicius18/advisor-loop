from spark_advisor.driver import (accumulate, build_result, classify_error,
                                   throughput_mbps, zero_metrics)


def test_zero_metrics_keys():
    z = zero_metrics()
    assert z["shuffle_write_bytes"] == 0 and z["gc_time_ms"] == 0
    assert set(z) >= {"input_bytes", "output_bytes", "shuffle_read_bytes",
                      "shuffle_write_bytes", "gc_time_ms", "memory_spill_bytes",
                      "disk_spill_bytes", "records_read"}


def test_accumulate_sums_stages_and_tolerates_missing():
    acc = zero_metrics()
    accumulate(acc, {"shuffle_write_bytes": 100, "gc_time_ms": 5, "input_bytes": None})
    accumulate(acc, {"shuffle_write_bytes": 50, "gc_time_ms": 3})
    assert acc["shuffle_write_bytes"] == 150 and acc["gc_time_ms"] == 8
    assert acc["input_bytes"] == 0   # None e ausente contam como 0


def test_accumulate_work_and_counts():
    from spark_advisor.driver import accumulate
    acc = zero_metrics()
    accumulate(acc, {"executor_cpu_time_ms": 1500, "executor_run_time_ms": 2000,
                     "num_tasks": 40, "num_stages": 1})
    accumulate(acc, {"executor_cpu_time_ms": 500, "num_tasks": 10, "num_stages": 1})
    assert acc["executor_cpu_time_ms"] == 2000 and acc["executor_run_time_ms"] == 2000
    assert acc["num_tasks"] == 50 and acc["num_stages"] == 2


def test_task_stats_max_and_median():
    from spark_advisor.driver import task_stats
    assert task_stats([]) == {"max_task_ms": 0, "median_task_ms": 0}
    assert task_stats([100, 300, 200, 5000]) == {"max_task_ms": 5000, "median_task_ms": 250}
    assert task_stats([7, 1, 3]) == {"max_task_ms": 7, "median_task_ms": 3}


_PLAN = """AdaptiveSparkPlan isFinalPlan=true
+- == Final Plan ==
   *(5) HashAggregate(keys=[d_year#1], functions=[sum(x)])
   +- AQEShuffleRead coalesced
      +- ShuffleQueryStage 2
         +- Exchange hashpartitioning(d_year#1, 200), ENSURE_REQUIREMENTS
            +- *(4) HashAggregate(keys=[d_year#1], functions=[partial_sum(x)])
               +- *(4) Project [d_year#1]
                  +- *(4) BroadcastHashJoin [ss_sold_date_sk#3], [d_date_sk#4], Inner
                     :- *(4) Sort [ss_item_sk#5 ASC], false, 0
                     :  +- SortMergeJoin [ss_item_sk#5], [i_item_sk#6], Inner
                     :     :- *(2) Filter isnotnull(ss_item_sk#5)
                     :     :  +- *(2) Scan parquet spark_catalog.default.store_sales
                     :     +- BroadcastExchange HashedRelationBroadcastMode
                     :        +- *(3) FileScan parquet default.item
                     +- BroadcastQueryStage 1
                        +- BroadcastExchange HashedRelationBroadcastMode
                           +- *(1) Scan parquet spark_catalog.default.date_dim
"""


def test_plan_summary_counts_operators_by_exact_name():
    from spark_advisor.driver import plan_summary
    s = plan_summary(_PLAN)
    assert s["plan_joins"] == 2 and s["plan_broadcast_joins"] == 1
    assert s["plan_sort_merge_joins"] == 1
    assert s["plan_exchanges"] == 1          # BroadcastExchange NAO conta como shuffle
    assert s["plan_aggregates"] == 2
    assert s["plan_sorts"] == 1              # SortMergeJoin/SortAggregate nao contam
    assert s["plan_scans"] == 3
    assert s["plan_lines"] > 10
    assert plan_summary("")["plan_lines"] == 0


def test_throughput():
    assert throughput_mbps(10_000_000, 1000.0) == 10.0   # 10MB em 1s = 10 MB/s
    assert throughput_mbps(0, 0) == 0.0


def test_classify_error():
    assert classify_error("java.lang.OutOfMemoryError: Java heap space") == "oom"
    assert classify_error("ExecutorLostFailure ... exit code 137") == "oom"
    assert classify_error("some other failure") == "failed"


def test_build_result():
    r = build_result("success", 1200.0, {"input_bytes": 5}, None)
    assert r["status"] == "success" and r["duration_ms"] == 1200.0 and r["metrics"]["input_bytes"] == 5
