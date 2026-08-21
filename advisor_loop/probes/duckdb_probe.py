import duckdb
import os
import re
import threading


class ProbeTimeout(Exception):
    pass


class DuckDBProbe:
    """Sondagem leve sobre tabelas parquet/delta. Paths locais ou s3:// (via httpfs/delta ext)."""
    def __init__(self, tables: dict[str, str], timeout_s: int = 10,
                 s3_config: dict | None = None):
        self.tables = tables
        self.con = duckdb.connect()
        self.timeout_s = timeout_s
        if s3_config:
            # AWS_EC2_METADATA_DISABLED evita que o delta kernel tente o
            # endpoint de metadata EC2 (trava fora de instancias AWS).
            os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
            os.environ["AWS_ALLOW_HTTP"] = "true"
            self.con.execute("INSTALL httpfs; LOAD httpfs; INSTALL delta; LOAD delta")
            use_ssl = "true" if s3_config.get("use_ssl", False) else "false"
            url_style = s3_config.get("url_style", "path")
            self.con.execute(
                "CREATE OR REPLACE SECRET minio (TYPE S3, "
                f"KEY_ID '{s3_config['key_id']}', SECRET '{s3_config['secret']}', "
                f"ENDPOINT '{s3_config['endpoint']}', URL_STYLE '{url_style}', "
                f"USE_SSL {use_ssl})"
            )

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
