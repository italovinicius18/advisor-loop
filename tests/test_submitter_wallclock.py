"""Teto de wall clock do submitter escala com tries em sessao."""
from types import SimpleNamespace
from advisor_loop.backends.spark_submitter import SparkSubmitter
from advisor_loop.episode import ConfigVector

CFG = ConfigVector(executor_cores=8, executor_memory_gb=28, executor_instances=6,
                   driver_memory_gb=8, shuffle_partitions=200, advisory_partition_mb=64,
                   aqe_enabled=True, broadcast_threshold_mb=10)


def _runner_never_terminal(calls):
    def run(args, capture_output=True, text=True):
        cmd = args[-1]
        calls.append(cmd)
        if "get pod" in cmd:
            return SimpleNamespace(stdout="Running", returncode=0)
        return SimpleNamespace(stdout="", returncode=0)
    return run


def test_error_excerpt_keeps_error_lines_not_boot_warnings():
    from advisor_loop.backends.spark_submitter import error_excerpt
    logs = "\n".join(["WARNING: Using incubator modules", "26/09/12 WARN NativeCodeLoader ...",
                      "Using Spark's default log4j profile"] + ["INFO x"] * 50 +
                     ["java.lang.OutOfMemoryError: Java heap space",
                      "ERROR SparkContext: Error initializing SparkContext.",
                      "Exception in thread main ... exit code 137"])
    ex = error_excerpt(logs)
    assert "OutOfMemoryError" in ex and "exit code 137" in ex
    assert "incubator" not in ex and "INFO x" not in ex
    assert error_excerpt("so info\nnada de erro") == "so info\nnada de erro"


def test_timeout_budget_scales_with_tries(monkeypatch):
    calls = []
    sub = SparkSubmitter(ssh_cmd=["bash", "-c"], image="img", poll_interval_s=100,
                         max_wall_clock_s=300, runner=_runner_never_terminal(calls))
    monkeypatch.setattr("advisor_loop.backends.spark_submitter.time.sleep", lambda s: None)
    r1 = sub.submit(CFG, "sf100", "q23", "r1", tries=1)
    polls1 = sum("get pod" in c for c in calls)
    calls.clear()
    r3 = sub.submit(CFG, "sf100", "q23", "r3", tries=3)
    polls3 = sum("get pod" in c for c in calls)
    assert r1["status"] == "failed" and "timeout" in r1["error"] and ">600s" in r1["error"]
    assert r3["status"] == "failed" and ">1200s" in r3["error"]
    assert polls3 > polls1                     # tries=3 -> orcamento (3+1)x, mais polls
