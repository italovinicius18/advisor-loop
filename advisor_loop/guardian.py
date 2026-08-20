from advisor_loop.episode import ConfigVector

class Guardian:
    def __init__(self, cluster: dict, infeasible: list[dict] | None = None):
        self.cluster = cluster
        self.infeasible = infeasible or []

    def validate(self, cfg: ConfigVector) -> tuple[bool, str]:
        c = self.cluster
        if cfg.executor_cores > c["max_cores_per_container"]:
            return False, f"executor_cores {cfg.executor_cores} > max_cores_per_container"
        if cfg.executor_memory_gb > c["max_memory_gb_per_container"]:
            return False, "executor_memory_gb acima do maximo por container"
        if cfg.executor_instances + 1 > c["max_pods"]:
            return False, "executors + driver excedem max_pods"
        if cfg.executor_cores * cfg.executor_instances > c["total_cores"]:
            return False, f"total_cores excedido ({cfg.executor_cores * cfg.executor_instances})"
        if cfg.executor_memory_gb * cfg.executor_instances + cfg.driver_memory_gb > c["total_memory_gb"]:
            return False, "memoria total excedida"
        d = cfg.model_dump()
        for known in self.infeasible:
            if all(d.get(k) == v for k, v in known.items()):
                return False, f"config sabidamente inviavel: {known}"
        return True, "ok"
