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
        # Overhead de memoria EXPLICITO (10%). Sem isto o Spark on K8s aplica o
        # fator de jobs nao-JVM (PySpark) = 40%: um executor de 96g pedia limite
        # de 137625Mi e a LimitRange do namespace (max 128Gi/container) recusava
        # o pod com Forbidden -- foi isso, nao OOM, que matou toda proposta de 96g
        # (rodada v3 e o replay sf10 q23). A matriz usou 8g fixos (96g -> 104Gi);
        # 10% da o mesmo teto (105.6Gi) e o padrao JVM nos demais tamanhos. Os
        # executores nao rodam Python (SQL puro), entao 40% nao tinha funcao.
        "spark.executor.memoryOverheadFactor": "0.1",
        "spark.driver.memoryOverheadFactor": "0.1",
    }
    out = []
    for k in pairs:  # dict preserva ordem de inserção → determinístico
        out += ["--conf", f"{k}={pairs[k]}"]
    return out
