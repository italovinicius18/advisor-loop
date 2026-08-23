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
