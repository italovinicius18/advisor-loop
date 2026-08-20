import duckdb


class DuckDBProbe:
    """Sondagem leve sobre tabelas parquet/delta. Paths locais ou s3:// (via httpfs/delta ext)."""
    def __init__(self, tables: dict[str, str]):
        self.tables = tables
        self.con = duckdb.connect()

    def _rel(self, table: str) -> str:
        path = self.tables[table]
        fn = "delta_scan" if not path.endswith(".parquet") else "read_parquet"
        return f"{fn}('{path}')"

    def list_tables(self) -> list[str]:
        return sorted(self.tables)

    def table_stats(self, table: str) -> dict:
        rc = self.con.sql(f"SELECT count(*) FROM {self._rel(table)}").fetchone()[0]
        cols = self.con.sql(f"DESCRIBE SELECT * FROM {self._rel(table)} LIMIT 0").fetchall()
        return {"row_count": rc, "columns": [{"name": c[0], "type": c[1]} for c in cols]}

    def sample_query(self, sql: str) -> list[tuple]:
        for name in self.tables:
            sql = sql.replace(name, self._rel(name))
        return self.con.sql(sql).fetchall()
