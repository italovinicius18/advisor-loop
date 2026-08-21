import json
import re
import urllib.error
import urllib.request

from advisor_loop.probes.duckdb_probe import ProbeQueryError


class RemoteDuckDBProbe:
    """Same interface as DuckDBProbe, but delegates query execution to an
    in-cluster DuckDB HTTP service (service/duckdb_probe_server.py) instead
    of running duckdb locally over the SSH tunnel. Compute moves to where the
    data (MinIO) lives; only small SQL requests and small JSON results cross
    the tunnel. BudgetedProbe wraps this identically to the local probe."""

    def __init__(self, base_url: str, tables: dict[str, str], timeout_s: int = 35):
        self.base_url = base_url.rstrip("/")
        self.tables = tables
        self.timeout_s = timeout_s

    def _rel(self, table: str) -> str:
        path = self.tables[table]
        fn = "delta_scan" if not path.endswith(".parquet") else "read_parquet"
        return f"{fn}('{path}')"

    def _post(self, sql: str) -> dict:
        body = json.dumps({"sql": sql}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/query", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise ProbeQueryError(str(e)) from e
        if "error" in payload:
            raise ProbeQueryError(payload["error"])
        return payload

    def list_tables(self) -> list[str]:
        return sorted(self.tables)

    def table_stats(self, table: str) -> dict:
        rel = self._rel(table)  # raises KeyError if table not in self.tables
        rc = self._post(f"SELECT count(*) FROM {rel}")["rows"][0][0]
        cols = self._post(f"DESCRIBE SELECT * FROM {rel} LIMIT 0")["rows"]
        return {"row_count": rc, "columns": [{"name": c[0], "type": c[1]} for c in cols]}

    def sample_query(self, sql: str) -> list[tuple]:
        for name in sorted(self.tables, key=len, reverse=True):
            sql = re.sub(rf"\b{re.escape(name)}\b", self._rel(name), sql)
        rows = self._post(sql)["rows"]
        return [tuple(r) for r in rows]
