from advisor_loop.episode import ConfigVector

def spark_confs(c: ConfigVector) -> list[str]:
    bcast = "-1" if c.broadcast_threshold_mb == -1 else f"{c.broadcast_threshold_mb}m"
    pairs = {
        "spark.executor.cores": c.executor_cores,
        "spark.executor.memory": f"{c.executor_memory_gb}g",
        "spark.executor.instances": c.executor_instances,
        "spark.driver.memory": f"{c.driver_memory_gb}g",
        "spark.sql.shuffle.partitions": c.shuffle_partitions,
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": f"{c.advisory_partition_mb}m",
        "spark.sql.adaptive.enabled": str(c.aqe_enabled).lower(),
        "spark.sql.autoBroadcastJoinThreshold": bcast,
    }
    out = []
    for k in pairs:  # dict preserva ordem de inserção → determinístico
        out += ["--conf", f"{k}={pairs[k]}"]
    return out
