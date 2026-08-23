from advisor_loop.backends.spark_k8s import SparkK8sBackend
from advisor_loop.episode import Episode, Recommendation, ConfigVector


class FakeSubmitter:
    def __init__(self, result):
        self.result = result
        self.seen = []

    def submit(self, config, sf, query, run_id):
        self.seen.append((sf, query, run_id))
        return self.result


def _rec():
    return Recommendation(rationale="", config=ConfigVector(
        executor_cores=8, executor_memory_gb=28, executor_instances=6, driver_memory_gb=8,
        shuffle_partitions=200, advisory_partition_mb=64, aqe_enabled=True, broadcast_threshold_mb=10))


def test_execute_success_maps_outcome():
    sub = FakeSubmitter({"status":"success","duration_ms":4300.0,"metrics":{"input_bytes":9},"error":None})
    be = SparkK8sBackend(sub, run_id_fn=lambda: "rid")
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    out = be.execute(ep, _rec())
    assert out.status == "completed"
    assert out.per_query["q03"].duration_ms == 4300.0
    assert out.per_query["q03"].metrics["input_bytes"] == 9
    assert sub.seen == [("sf1","q03","rid")]


def test_execute_oom_propagates():
    sub = FakeSubmitter({"status":"oom","duration_ms":None,"metrics":{},"error":"OOM"})
    be = SparkK8sBackend(sub, run_id_fn=lambda: "rid")
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    out = be.execute(ep, _rec())
    assert out.status == "oom" and out.per_query["q03"].status == "oom"


def test_execute_failed_query_maps_outcome_to_error_not_crash():
    sub = FakeSubmitter({"status":"failed","duration_ms":None,"metrics":{},"error":"boom"})
    be = SparkK8sBackend(sub, run_id_fn=lambda: "rid")
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    out = be.execute(ep, _rec())
    assert out.status == "error"
    assert out.per_query["q03"].status == "failed"


def test_execute_timeout_query_maps_outcome_to_error():
    sub = FakeSubmitter({"status":"timeout","duration_ms":None,"metrics":{},"error":"timeout"})
    be = SparkK8sBackend(sub, run_id_fn=lambda: "rid")
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    out = be.execute(ep, _rec())
    assert out.status == "error"
    assert out.per_query["q03"].status == "timeout"
