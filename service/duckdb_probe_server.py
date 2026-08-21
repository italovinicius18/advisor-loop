"""In-cluster DuckDB probe service.

Runs a minimal HTTP server next to MinIO (namespace italo-unb) so the
advisor's data-probing sends small SQL strings and gets back small JSON
results instead of streaming data over the SSH tunnel used by the local
DuckDBProbe path. Deliberately stdlib-only (http.server, no FastAPI/uvicorn)
to keep the image tiny and dependency-light - the only real dependency is
duckdb itself.

Endpoints:
  GET  /health -> {"status": "ok"}
  POST /query  {"sql": "..."} -> {"columns": [...], "rows": [[...], ...]}
               or {"error": "..."} (always HTTP 200 - the client turns an
               "error" field into a ProbeQueryError, never a raw exception).

Run directly: python service/duckdb_probe_server.py
"""
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import duckdb

DEFAULT_QUERY_TIMEOUT_S = int(os.environ.get("QUERY_TIMEOUT_S", "30"))

# Defense in depth: the BudgetedProbe layer already enforces SELECT/WITH-only
# + single-statement client-side, but the server re-checks independently
# since it is the actual trust boundary. DESCRIBE is allowed alongside
# SELECT/WITH because table_stats() needs it (DESCRIBE SELECT ... LIMIT 0)
# and it is just as read-only as a SELECT.
_ALLOWED_PREFIX = re.compile(r"^(select|with|describe)\b", re.I)


def _validate_readonly(sql: str) -> str | None:
    """Return an error message if sql isn't a safe read-only single
    statement, else None."""
    stripped = re.sub(r"--[^\n]*", "", sql)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.S)
    stripped = stripped.strip().rstrip(";").strip()

    if ";" in stripped:
        return "apenas um statement por probe"
    if not _ALLOWED_PREFIX.match(stripped):
        return "only SELECT/WITH allowed"
    return None


def run_query(con, sql: str, timeout_s: int = DEFAULT_QUERY_TIMEOUT_S) -> dict:
    """Execute sql on con, returning {"columns": ..., "rows": ...} or
    {"error": ...}. Never raises - callers (the HTTP handler) always get a
    JSON-serializable dict back."""
    guard_err = _validate_readonly(sql)
    if guard_err:
        return {"error": guard_err}

    result: dict = {}
    exception: list[Exception] = []

    def execute():
        try:
            rel = con.sql(sql)
            cols = [d[0] for d in rel.description]
            rows = [list(r) for r in rel.fetchall()]
            result["columns"] = cols
            result["rows"] = rows
        except Exception as e:
            exception.append(e)

    thread = threading.Thread(target=execute)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        con.interrupt()
        thread.join(timeout=1)
        return {"error": f"query exceeded {timeout_s}s timeout"}

    if exception:
        return {"error": str(exception[0])}

    return result


def make_handler(con, timeout_s: int = DEFAULT_QUERY_TIMEOUT_S):
    """Build a BaseHTTPRequestHandler class bound to `con`. A factory (not a
    module-level class) so tests can drive the handler over a local duckdb
    connection/parquet fixture without real MinIO, and so the real server can
    bind it to the MinIO-backed connection built by build_app()."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
            pass  # quiet by default; rely on k8s pod logs for real traffic if needed

        def _send_json(self, status: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/query":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(body)
                sql = payload["sql"]
            except Exception:
                self._send_json(200, {"error": "invalid request body"})
                return
            self._send_json(200, run_query(con, sql, timeout_s))

    return Handler


def build_app():
    """Build the real MinIO-backed duckdb connection for the in-cluster
    service."""
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    os.environ["AWS_ALLOW_HTTP"] = "true"

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL delta; LOAD delta")

    endpoint = os.environ.get("MINIO_ENDPOINT", "minio.italo-unb.svc.cluster.local:9000")
    key_id = os.environ.get("MINIO_KEY", "minioadmin")
    secret = os.environ.get("MINIO_SECRET", "minioadmin")
    con.execute(
        "CREATE OR REPLACE SECRET minio (TYPE S3, "
        f"KEY_ID '{key_id}', SECRET '{secret}', "
        f"ENDPOINT '{endpoint}', URL_STYLE 'path', USE_SSL false)"
    )
    return con


def main():
    con = build_app()
    port = int(os.environ.get("PORT", "8000"))
    timeout_s = int(os.environ.get("QUERY_TIMEOUT_S", str(DEFAULT_QUERY_TIMEOUT_S)))
    handler = make_handler(con, timeout_s)
    server = HTTPServer(("0.0.0.0", port), handler)
    print(f"duckdb-probe-service listening on :{port} (query timeout {timeout_s}s)")
    server.serve_forever()


if __name__ == "__main__":
    main()
