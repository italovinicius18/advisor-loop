import duckdb
import re
import threading


class ProbeTimeout(Exception):
    pass


class DuckDBProbe:
    """Sondagem leve sobre tabelas parquet/delta. Paths locais ou s3:// (via httpfs/delta ext)."""
    def __init__(self, tables: dict[str, str], timeout_s: int = 10):
        self.tables = tables
        self.con = duckdb.connect()
        self.timeout_s = timeout_s

    def _rel(self, table: str) -> str:
        path = self.tables[table]
        fn = "delta_scan" if not path.endswith(".parquet") else "read_parquet"
        return f"{fn}('{path}')"

    def _run(self, sql: str) -> list[tuple]:
        """Execute SQL with timeout enforcement."""
        result = []
        exception = []

        def execute():
            try:
                result.extend(self.con.sql(sql).fetchall())
            except Exception as e:
                exception.append(e)

        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout_s)

        if thread.is_alive():
            self.con.interrupt()
            thread.join(timeout=1)
            raise ProbeTimeout(f"Query exceeded {self.timeout_s}s timeout")

        if exception:
            raise exception[0]

        return result

    def list_tables(self) -> list[str]:
        return sorted(self.tables)

    def table_stats(self, table: str) -> dict:
        rc = self._run(f"SELECT count(*) FROM {self._rel(table)}")[0][0]
        cols = self._run(f"DESCRIBE SELECT * FROM {self._rel(table)} LIMIT 0")
        return {"row_count": rc, "columns": [{"name": c[0], "type": c[1]} for c in cols]}

    def sample_query(self, sql: str) -> list[tuple]:
        for name in sorted(self.tables, key=len, reverse=True):
            sql = re.sub(rf"\b{re.escape(name)}\b", self._rel(name), sql)
        return self._run(sql)
