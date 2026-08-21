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
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import duckdb

DEFAULT_QUERY_TIMEOUT_S = int(os.environ.get("QUERY_TIMEOUT_S", "30"))
# Bound on the post-interrupt join, see run_query().
_INTERRUPT_JOIN_TIMEOUT_S = 30

# Defense in depth: the BudgetedProbe layer already enforces SELECT/WITH-only
# + single-statement client-side, but the server re-checks independently
# since it is the actual trust boundary. DESCRIBE is allowed alongside
# SELECT/WITH because table_stats() needs it (DESCRIBE SELECT ... LIMIT 0)
# and it is just as read-only as a SELECT.
_ALLOWED_PREFIX = re.compile(r"^(select|with|describe)\b", re.I)

# SELECT/WITH-only is NOT enough: duckdb table functions callable inside a
# SELECT can read pod-local files (read_text/read_csv/read_json/glob) or make
# arbitrary outbound HTTP requests via httpfs (SSRF). A pod running in-cluster
# makes the local filesystem alone worth blocking (e.g. a mounted k8s SA
# token).
#
# Round 1 blacklisted "http://" as text - defeated by string concatenation
# ('http:' || '//169.254.169.254/...' never contains that literal substring).
# Round 2 blacklisted the *builder functions* (concat/printf/...) instead -
# defeated by replace()/regexp_replace(), which weren't on the list. The
# lesson: duckdb's scalar function surface is large and blacklisting it is
# open-ended - there is always one more string-building function.
#
# Round 3 (this one) INVERTS to a WHITELIST on the one thing that actually
# matters: the legitimate probe SQL (DuckDBProbe/RemoteDuckDBProbe) only
# ever calls an S3 reader with a single bare 's3://...' literal - nothing
# built, nothing else in the argument list. So instead of trying to
# enumerate every way a hostile argument could be constructed, require the
# call to be in EXACTLY that shape:
#   1. every occurrence of an S3-capable reader function name (delta_scan,
#      read_parquet, parquet_scan, parquet_metadata, parquet_schema) must be
#      immediately followed by `(<optional ws>'s3://...'<optional ws>)` -
#      a single bare quoted s3:// literal and nothing else. Anything else in
#      the argument position (a function call, a list, a concatenation, a
#      non-s3 literal) is rejected - it no longer matters WHAT scalar
#      function might have built the argument, because no argument other
#      than a bare literal is accepted at all.
#   2. local-file reader functions - read_text/read_blob/read_csv/read_json/
#      read_ndjson/glob AND their real duckdb variants (read_csv_auto,
#      read_json_auto, read_ndjson_auto, read_json_objects, sniff_csv) - are
#      still rejected outright, in any usage. This regex layer is genuine
#      defense-in-depth, not the only thing stopping them: `SET
#      disabled_filesystems='LocalFileSystem'` in build_app() blocks local
#      reads at the engine level regardless of what slips past the regex,
#      and any of these called with an http(s):// argument instead of a
#      local path would separately be caught by the literal-scheme scan
#      (rule 3 below) even if a variant were missing from this list.
#   3. the string-builder blacklist (||, concat, printf, ...) and the
#      "any URL-shaped literal must be s3://" literal scan from round 2 are
#      KEPT as defense-in-depth, but neither is load-bearing any more - rule
#      1 alone already blocks every bypass found across all three rounds.
_S3_READER_FUNCS = ("delta_scan", "read_parquet", "parquet_scan", "parquet_metadata", "parquet_schema")
_LOCAL_FILE_FUNCS = (
    "read_text", "read_blob",
    "read_csv_auto", "read_csv",
    "read_json_objects", "read_json_auto", "read_json",
    "read_ndjson_auto", "read_ndjson",
    "sniff_csv", "glob",
)
_STRING_BUILDER_FUNCS = (
    "concat_ws", "concat", "printf", "format", "chr", "to_hex", "from_hex",
    "list_aggr", "string_agg", "repeat", "reverse", "translate", "encode", "decode",
)
_LOCAL_FILE_FUNC_RE = re.compile(r"\b(" + "|".join(_LOCAL_FILE_FUNCS) + r")\s*\(", re.I)
_STRING_BUILDER_FUNC_RE = re.compile(r"\b(" + "|".join(_STRING_BUILDER_FUNCS) + r")\s*\(", re.I)
_S3_READER_CALL_RE = re.compile(r"\b(" + "|".join(_S3_READER_FUNCS) + r")\s*\(", re.I)
# The one allowed shape of an S3 reader call's argument list: a single bare
# 's3://...' literal, nothing else.
_BARE_S3_LITERAL_ARG_RE = re.compile(r"\s*'s3://[^']*'\s*\)", re.I)
# A single-quoted SQL string literal, honoring the '' (doubled-quote) escape
# for an embedded literal quote.
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")


def _s3_readers_use_bare_literal_only(stripped: str) -> str | None:
    """Every S3-capable reader call must be `name('s3://...')` exactly - a
    single bare literal argument, nothing built, nothing else. Returns an
    error message if any call doesn't match that shape, else None."""
    for call in _S3_READER_CALL_RE.finditer(stripped):
        if not _BARE_S3_LITERAL_ARG_RE.match(stripped, call.end()):
            return "forbidden function/URL"
    return None


def _strip_sql(sql: str) -> str:
    """Comment-strip + trim exactly once. The guard validates this string
    AND run_query executes this same string (not the original) - "what was
    validated is what runs", no divergence between the two."""
    stripped = re.sub(r"--[^\n]*", "", sql)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.S)
    return stripped.strip().rstrip(";").strip()


def _check_readonly(stripped: str) -> str | None:
    """Guard checks against an already-stripped SQL string (see
    _strip_sql). Returns an error message if unsafe, else None."""
    if ";" in stripped:
        return "apenas um statement por probe"
    if not _ALLOWED_PREFIX.match(stripped):
        return "only SELECT/WITH allowed"

    if _LOCAL_FILE_FUNC_RE.search(stripped):
        return "forbidden function/URL"

    err = _s3_readers_use_bare_literal_only(stripped)
    if err:
        return err

    # Defense-in-depth, not load-bearing (see comment above): kept in case a
    # future reader function is added to _S3_READER_FUNCS late, or a probe
    # shape we haven't anticipated slips past rule 1 above.
    if "||" in stripped:
        return "forbidden function/URL"
    if _STRING_BUILDER_FUNC_RE.search(stripped):
        return "forbidden function/URL"
    for literal in _STRING_LITERAL_RE.findall(stripped):
        value = literal[1:-1].replace("''", "'")
        if "://" in value and not value.lower().startswith("s3://"):
            return "forbidden function/URL"

    return None


def _validate_readonly(sql: str) -> str | None:
    """Return an error message if sql isn't a safe read-only single
    statement, else None. Public entry point (used directly by tests and
    indirectly via run_query, which validates and executes the same
    stripped string)."""
    return _check_readonly(_strip_sql(sql))


def run_query(con, sql: str, timeout_s: int = DEFAULT_QUERY_TIMEOUT_S) -> dict:
    """Execute sql on con, returning {"columns": ..., "rows": ...} or
    {"error": ...}. Never raises - callers (the HTTP handler) always get a
    JSON-serializable dict back."""
    stripped = _strip_sql(sql)
    guard_err = _check_readonly(stripped)
    if guard_err:
        return {"error": guard_err}

    result: dict = {}
    exception: list[Exception] = []

    def execute():
        try:
            # Execute the SAME string the guard validated, not the original
            # `sql` - guarantees "what was validated is what runs" with no
            # divergence between the two (the client never sends comments,
            # so this is a no-op on legitimate traffic).
            rel = con.sql(stripped)
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
        # Bounded join, not uncapped: the caller holds query_lock for the
        # duration of this call, and con is not safe for concurrent use, so
        # returning while the worker might still be inside con.sql() would
        # let the *next* query race on the same connection. con.interrupt()
        # normally makes duckdb abort promptly - but if the worker is stuck
        # inside a call with no interrupt point (e.g. a wedged httpfs
        # socket), an UNCAPPED join would hang forever holding the lock,
        # starving every future /query. Bound it instead: if the worker is
        # still alive after _INTERRUPT_JOIN_TIMEOUT_S, the connection is
        # unrecoverable under the lock - hard-exit so k8s restarts the pod
        # (self-heal). /health is unaffected either way (it never touches
        # query_lock or con).
        thread.join(timeout=_INTERRUPT_JOIN_TIMEOUT_S)
        if thread.is_alive():
            print(
                f"FATAL: duckdb connection unresponsive {_INTERRUPT_JOIN_TIMEOUT_S}s "
                "after interrupt() - exiting for pod restart", file=sys.stderr,
            )
            os._exit(1)
        return {"error": f"query exceeded {timeout_s}s timeout"}

    if exception:
        return {"error": str(exception[0])}

    return result


def make_handler(con, timeout_s: int = DEFAULT_QUERY_TIMEOUT_S):
    """Build a BaseHTTPRequestHandler class bound to `con`. A factory (not a
    module-level class) so tests can drive the handler over a local duckdb
    connection/parquet fixture without real MinIO, and so the real server can
    bind it to the MinIO-backed connection built by build_app()."""

    # A duckdb.Connection is not safe for concurrent use from multiple
    # threads. ThreadingHTTPServer means two overlapping POST /query calls
    # would otherwise race on the same shared `con`. Serialize query
    # execution with a lock scoped to this connection; GET /health never
    # touches `con` or this lock, so health checks stay responsive even
    # while a query is in flight or queued behind the lock.
    query_lock = threading.Lock()

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
            with query_lock:
                result = run_query(con, sql, timeout_s)
            self._send_json(200, result)

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

    # Engine-level backstop for the SQL guard above: even if a query text
    # slips past the regex, duckdb itself refuses any local-file access
    # (read_text/read_csv/glob on '/...'), including the mounted k8s SA
    # token path. S3/httpfs access (delta_scan/read_parquet on s3://...) is
    # unaffected - only the *local* filesystem is disabled.
    con.execute("SET disabled_filesystems='LocalFileSystem'")
    return con


def main():
    con = build_app()
    port = int(os.environ.get("PORT", "8000"))
    timeout_s = int(os.environ.get("QUERY_TIMEOUT_S", str(DEFAULT_QUERY_TIMEOUT_S)))
    handler = make_handler(con, timeout_s)
    # ThreadingHTTPServer, not HTTPServer: a single-threaded server would let
    # a slow/heavy query block /health (and every other in-flight request)
    # for its full duration.
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"duckdb-probe-service listening on :{port} (query timeout {timeout_s}s)")
    server.serve_forever()


if __name__ == "__main__":
    main()
