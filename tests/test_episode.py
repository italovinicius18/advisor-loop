from advisor_loop.episode import ConfigVector, task_key
import pydantic, pytest

def test_config_vector_bounds():
    cv = ConfigVector(executor_cores=4, executor_memory_gb=14, executor_instances=4,
                      driver_memory_gb=8, shuffle_partitions=400,
                      advisory_partition_mb=64, aqe_enabled=True, broadcast_threshold_mb=10)
    assert cv.executor_cores == 4
    with pytest.raises(pydantic.ValidationError):
        ConfigVector(executor_cores=99, executor_memory_gb=14, executor_instances=4,
                     driver_memory_gb=8, shuffle_partitions=400,
                     advisory_partition_mb=64, aqe_enabled=True, broadcast_threshold_mb=10)

def test_config_hash_stable_and_task_key():
    kw = dict(executor_cores=4, executor_memory_gb=14, executor_instances=4,
              driver_memory_gb=8, shuffle_partitions=400, advisory_partition_mb=64,
              aqe_enabled=True, broadcast_threshold_mb=10)
    assert ConfigVector(**kw).config_hash() == ConfigVector(**kw).config_hash()
    k = task_key("w", "d100", "q17", ConfigVector(**kw).config_hash(), 2)
    assert len(k) == 16
