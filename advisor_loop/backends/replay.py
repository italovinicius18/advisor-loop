from pathlib import Path
import duckdb
from advisor_loop.episode import Episode, Outcome, QueryResult, Recommendation

class ReplayBackend:
    def __init__(self, reference: str | Path):
        self.con = duckdb.connect()
        self.con.sql(f"CREATE VIEW ref AS SELECT * FROM read_parquet('{reference}')")

    def execute(self, ep: Episode, rec: Recommendation) -> Outcome:
        c = rec.config
        per_query, total = {}, 0.0
        for q in ep.queries:
            rows = self.con.execute(
                "SELECT status, duration_ms FROM ref WHERE dataset=? AND query=? "
                "AND executor_cores=? AND executor_memory_gb=? AND executor_instances=?",
                [ep.dataset, q, c.executor_cores, c.executor_memory_gb, c.executor_instances],
            ).fetchall()
            if not rows:
                return Outcome(status="infeasible", per_query={},
                               wall_clock_s=0.0, cost_usd=0.0)
            status, dur = rows[0]
            dur = float(dur) if dur is not None else None
            per_query[q] = QueryResult(status=status, duration_ms=dur)
            total += (dur or 0.0) / 1000
        return Outcome(status="completed", per_query=per_query, wall_clock_s=total)

    def oracle(self, dataset: str, query: str) -> float | None:
        row = self.con.execute(
            "SELECT min(duration_ms) FROM ref WHERE dataset=? AND query=? AND status='success'",
            [dataset, query]).fetchone()
        return float(row[0]) if row[0] is not None else None
