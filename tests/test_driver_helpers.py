from spark_advisor.driver import build_result, classify_error, throughput_mbps


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
