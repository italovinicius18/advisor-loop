from advisor_loop.backends.spark_conf import spark_confs
from advisor_loop.episode import ConfigVector

def _cv(**o):
    base = dict(executor_cores=8, executor_memory_gb=28, executor_instances=6,
                driver_memory_gb=8, shuffle_partitions=200, advisory_partition_mb=64,
                aqe_enabled=True, broadcast_threshold_mb=10)
    return ConfigVector(**{**base, **o})

def test_translation():
    c = spark_confs(_cv())
    s = " ".join(c)
    assert "spark.executor.cores=8" in s
    assert "spark.executor.memory=28g" in s
    assert "spark.executor.instances=6" in s
    assert "spark.driver.memory=8g" in s
    assert "spark.sql.shuffle.partitions=200" in s
    assert "spark.sql.adaptive.advisoryPartitionSizeInBytes=64m" in s
    assert "spark.sql.adaptive.enabled=true" in s
    assert "spark.sql.autoBroadcastJoinThreshold=10m" in s

def test_broadcast_off():
    assert "spark.sql.autoBroadcastJoinThreshold=-1" in " ".join(spark_confs(_cv(broadcast_threshold_mb=-1)))
