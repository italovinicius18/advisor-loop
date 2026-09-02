"""Static knob-cards + cluster caps injected into the advisor prompt (T1).

This is the Phase-2 context enrichment being piloted: instead of a discrete
menu, the LLM proposes freely but is grounded by (a) what each knob does and
which direction helps, and (b) the hard resource ceiling of this node -- the
Phase-1 weakness was that every LLM miss was an *infeasible* pick, so the cap
is stated explicitly.
"""

# This pilot runs on a single local[*] node (WSL, 28 cores / 15 GB, ~7 GB free
# for the JVM). executor_instances / executor_memory_gb are inert in local
# mode; the effective knobs are the six below.
CLUSTER_CAPS = {
    "node": "single local[*] Spark, 1 process",
    "max_executor_cores": 8,      # local[N], capped for headroom
    "max_driver_memory_gb": 6,    # JVM heap ceiling (~7 GB free)
    "note": "executor_instances and executor_memory_gb have no effect locally; "
            "executor_cores sets local[N] parallelism, driver_memory_gb sets the heap.",
}

KNOB_CARDS = {
    "executor_cores": "Parallelism (local[N]). More cores = more concurrent "
        "tasks, but past the physical core count or the data's partition count "
        "it stops helping and adds scheduling overhead. Cap 8 here.",
    "driver_memory_gb": "JVM heap for the single local process. Too low -> spill "
        "to disk and GC pressure; too high -> wasted / OOM the box. Cap 6 GB here.",
    "shuffle_partitions": "spark.sql.shuffle.partitions. Target partitions after a "
        "shuffle. Too few -> large partitions, spill, poor parallelism; too many "
        "-> tiny tasks and scheduling overhead. AQE coalesces down from this.",
    "advisory_partition_mb": "spark.sql.adaptive.advisoryPartitionSizeInBytes. "
        "Target post-shuffle partition size AQE coalesces toward. Smaller = more, "
        "smaller partitions; larger = fewer, bigger. Only bites when AQE is on.",
    "aqe_enabled": "spark.sql.adaptive.enabled. Runtime re-optimization: coalesces "
        "shuffle partitions, switches join strategies, handles skew. Usually on.",
    "broadcast_threshold_mb": "spark.sql.autoBroadcastJoinThreshold. Tables under "
        "this size are broadcast (no shuffle for that join). Higher -> more "
        "broadcasts (fast if the small side fits memory, OOM risk if not). "
        "-1 disables broadcast joins entirely. Matters most for multi-join queries.",
}
