from pathlib import Path
import duckdb
from advisor_loop.episode import Episode, Outcome, QueryResult, Recommendation

# Colunas de telemetria que, se presentes no parquet de referência, são
# expostas em QueryResult.metrics — insumo da ablação A3t (só-tempo) vs A3m
# (+telemetria) no histórico do advisor. Parquets antigos sem essas colunas
# continuam funcionando (metrics fica vazio).
_METRIC_COLS = [
    "shuffle_read_bytes", "shuffle_write_bytes", "input_bytes", "gc_time_ms",
    "num_stages", "num_tasks", "num_joins", "num_broadcast_joins",
]


class ReplayBackend:
    def __init__(self, reference: str | Path, emit_metrics: bool = True):
        # emit_metrics=False zera QueryResult.metrics mesmo com telemetria no
        # parquet -> serve o braço A3t (histórico só-tempo) da ablação.
        self.con = duckdb.connect()
        self.con.sql(f"CREATE VIEW ref AS SELECT * FROM read_parquet('{reference}')")
        cols = {r[0] for r in self.con.execute("DESCRIBE ref").fetchall()}
        self._metric_cols = [c for c in _METRIC_COLS if c in cols] if emit_metrics else []

    def execute(self, ep: Episode, rec: Recommendation) -> Outcome:
        c = rec.config
        sel = "status, duration_ms" + (
            ", " + ", ".join(self._metric_cols) if self._metric_cols else "")
        per_query, total = {}, 0.0
        for q in ep.queries:
            rows = self.con.execute(
                f"SELECT {sel} FROM ref WHERE dataset=? AND query=? "
                "AND executor_cores=? AND executor_memory_gb=? AND executor_instances=?",
                [ep.dataset, q, c.executor_cores, c.executor_memory_gb, c.executor_instances],
            ).fetchall()
            if not rows:
                return Outcome(status="infeasible", per_query={},
                               wall_clock_s=0.0, cost_usd=0.0)
            row = rows[0]
            status, dur = row[0], row[1]
            dur = float(dur) if dur is not None else None
            metrics = {c_: row[2 + i] for i, c_ in enumerate(self._metric_cols)
                       if row[2 + i] is not None}
            per_query[q] = QueryResult(status=status, duration_ms=dur, metrics=metrics)
            total += (dur or 0.0) / 1000
        return Outcome(status="completed", per_query=per_query, wall_clock_s=total)

    def oracle(self, dataset: str, query: str) -> float | None:
        row = self.con.execute(
            "SELECT min(duration_ms) FROM ref WHERE dataset=? AND query=? AND status='success'",
            [dataset, query]).fetchone()
        return float(row[0]) if row[0] is not None else None
