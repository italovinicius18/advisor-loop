from advisor_loop.guardian import Guardian
from advisor_loop.episode import ConfigVector

CLUSTER = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
           "max_pods": 20, "total_cores": 100, "total_memory_gb": 400}

def _cv(**over):
    base = dict(executor_cores=4, executor_memory_gb=14, executor_instances=4,
                driver_memory_gb=8, shuffle_partitions=400, advisory_partition_mb=64,
                aqe_enabled=True, broadcast_threshold_mb=10)
    return ConfigVector(**{**base, **over})

def test_ok_and_over_quota():
    g = Guardian(CLUSTER)
    assert g.validate(_cv())[0] is True
    ok, reason = g.validate(_cv(executor_cores=20, executor_instances=10))  # 200 > 100 cores
    assert ok is False and "total_cores" in reason

def test_known_infeasible():
    g = Guardian(CLUSTER, infeasible=[{"executor_memory_gb": 14, "executor_instances": 2}])
    ok, reason = g.validate(_cv(executor_instances=2))
    assert ok is False and "inviavel" in reason
