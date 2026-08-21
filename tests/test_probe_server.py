import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import duckdb
import pytest

from service.duckdb_probe_server import make_handler


@pytest.fixture
def running_server():
    """Real local http.server bound to an in-memory duckdb connection - no
    MinIO, no network extensions. Mirrors the in-cluster service shape."""
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT range AS id, 'x' AS name FROM range(10)")
    handler = make_handler(con, timeout_s=5)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def _post(base_url, sql):
    body = json.dumps({"sql": sql}).encode()
    req = urllib.request.Request(
        f"{base_url}/query", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def test_health(running_server):
    with urllib.request.urlopen(f"{running_server}/health", timeout=5) as resp:
        assert resp.status == 200
        assert json.loads(resp.read()) == {"status": "ok"}


def test_query_select_returns_columns_and_rows(running_server):
    payload = _post(running_server, "SELECT id, name FROM t ORDER BY id LIMIT 3")
    assert payload["columns"] == ["id", "name"]
    assert payload["rows"] == [[0, "x"], [1, "x"], [2, "x"]]


def test_query_with_cte_allowed(running_server):
    payload = _post(running_server, "WITH c AS (SELECT count(*) AS n FROM t) SELECT n FROM c")
    assert payload["rows"] == [[10]]


def test_query_describe_allowed_for_table_stats(running_server):
    """DESCRIBE is needed by table_stats() via the remote client - the
    read-only guard must not block it even though it isn't SELECT/WITH."""
    payload = _post(running_server, "DESCRIBE SELECT * FROM t LIMIT 0")
    assert "error" not in payload
    names = [row[0] for row in payload["rows"]]
    assert "id" in names and "name" in names


def test_query_rejects_non_select(running_server):
    payload = _post(running_server, "PRAGMA database_list")
    assert "error" in payload
    assert "SELECT/WITH" in payload["error"]


def test_query_rejects_second_statement(running_server):
    payload = _post(running_server, "SELECT 1; DROP TABLE t")
    assert "error" in payload


def test_query_duckdb_error_returns_200_with_error_not_crash(running_server):
    """A genuine duckdb error (e.g. binder error) must come back as HTTP 200
    with an 'error' field - not a 500 - so the client can turn it into a
    ProbeQueryError without special-casing status codes."""
    con_error_sql = "SELECT sum(name) FROM t LIMIT 1"
    body = json.dumps({"sql": con_error_sql}).encode()
    req = urllib.request.Request(
        f"{running_server}/query", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read())
    assert "error" in payload


def test_query_timeout_returns_error(running_server):
    con = duckdb.connect()
    con.execute("CREATE TABLE big AS SELECT range AS id FROM range(200000)")
    handler = make_handler(con, timeout_s=1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        payload = _post(
            base_url,
            "SELECT count(*) FROM big a, big b, big c LIMIT 1",
        )
        assert "error" in payload
        assert "timeout" in payload["error"].lower()
    finally:
        server.shutdown()
        thread.join(timeout=2)


# --- security: local-file / SSRF exfiltration must never execute ---------

def test_query_rejects_local_file_read_text(running_server):
    """SELECT-only isn't enough on its own - read_text() called inside a
    SELECT would read pod-local files (e.g. the mounted k8s SA token)."""
    payload = _post(running_server, "SELECT * FROM read_text('/etc/hostname')")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_local_csv_read(running_server):
    payload = _post(running_server, "SELECT * FROM read_csv('/etc/passwd')")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_glob(running_server):
    payload = _post(running_server, "SELECT * FROM glob('/etc/*')")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_http_url_ssrf(running_server):
    """httpfs makes read_csv('http://internal-svc') an SSRF primitive."""
    payload = _post(running_server, "SELECT * FROM read_csv('http://169.254.169.254/latest')")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_bare_https_url_token(running_server):
    payload = _post(running_server, "SELECT 'https://example.com' AS x")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_plain_select_still_allowed(running_server):
    """The forbidden-token guard must not false-positive on ordinary probe SQL."""
    payload = _post(running_server, "SELECT 42 AS x")
    assert "error" not in payload
    assert payload["rows"] == [[42]]


# --- security: SSRF closed STRUCTURALLY (allowlist the literal form) -----
#
# A text blacklist on "http://" is not enough: string concatenation builds
# the substring at query time without it ever appearing literally in the
# SQL text, e.g. 'http:' || '//169.254.169.254/...'. The guard instead (1)
# forbids any string-building construct (||, concat/printf/format/...) and
# (2) requires every URL-shaped string literal to start with s3://. Together
# these mean there is no way to assemble a non-s3 URL that reaches httpfs.

def test_query_rejects_concat_ssrf_bypass(running_server):
    """The exact reproduced bypass: string concatenation defeats a naive
    'http://' text blacklist but not the structural guard."""
    payload = _post(
        running_server,
        "SELECT * FROM read_parquet('http:' || '//169.254.169.254/x')",
    )
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_non_s3_url_literal_http(running_server):
    payload = _post(running_server, "SELECT * FROM read_parquet('http://evil/x.parquet')")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_non_s3_url_literal_file(running_server):
    payload = _post(running_server, "SELECT * FROM delta_scan('file:///etc/x')")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_glob_with_space_before_paren(running_server):
    """Whitespace-before-paren must not bypass the forbidden-function check."""
    payload = _post(running_server, "SELECT * FROM glob ('/etc/*')")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_printf_string_builder(running_server):
    payload = _post(running_server, "SELECT printf('%s','x')")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_guard_allows_legitimate_s3_probe_sql():
    """The real probe SQL shapes (bare s3:// literals, no concatenation, no
    string-builder functions) must still pass the guard. Checked directly
    against the guard function rather than executed end-to-end, since these
    queries need real MinIO/S3 connectivity to actually run."""
    from service.duckdb_probe_server import _validate_readonly

    assert _validate_readonly(
        "SELECT count(*) FROM delta_scan('s3://silver/sf1/delta/parquet/item')"
    ) is None
    assert _validate_readonly(
        "DESCRIBE SELECT * FROM read_parquet('s3://silver/sf1/delta/parquet/item') LIMIT 0"
    ) is None
    assert _validate_readonly(
        "SELECT ss_x FROM delta_scan('s3://silver/sf1/delta/parquet/store_sales') LIMIT 5"
    ) is None


# --- security round 3: SSRF closed by WHITELISTING the argument shape ----
#
# Round 2's builder-function blacklist was defeated by replace()/
# regexp_replace() - not on the list, and duckdb's scalar-function surface
# is too large to blacklist exhaustively (there's always one more). Round 3
# inverts to a whitelist: every S3-reader call (delta_scan/read_parquet/
# parquet_scan/parquet_metadata/parquet_schema) must be EXACTLY
# `name('s3://...')` - a single bare literal argument. Anything else in the
# argument position - a function call regardless of which one, a list, a
# concatenation, a non-s3 literal - is rejected. It no longer matters what
# scalar function might have built the argument, because no argument other
# than a bare literal is accepted at all.

def test_query_rejects_replace_ssrf_bypass(running_server):
    """The exact round-3 reproduced bypass: replace() reconstructs a clean
    URL and isn't on any builder blacklist - the whitelist rejects it anyway
    because the argument still isn't a bare literal."""
    payload = _post(
        running_server,
        "SELECT * FROM read_parquet(replace('http:XX//169.254.169.254/x','XX','//'))",
    )
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_regexp_replace_ssrf_bypass(running_server):
    payload = _post(
        running_server,
        "SELECT * FROM delta_scan(regexp_replace('http:','p','p://x'))",
    )
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_concat_on_s3_reader_even_with_s3_prefix(running_server):
    """Not a bare literal call - concatenation is rejected regardless of
    whether the pieces happen to spell something s3-looking."""
    payload = _post(running_server, "SELECT * FROM read_parquet('s3://a' || 'b')")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_list_argument_to_s3_reader(running_server):
    """A list literal isn't a bare quoted literal - rejected even though
    duckdb's readers accept a list-of-paths form in real usage."""
    payload = _post(running_server, "SELECT * FROM read_parquet(['s3://a'])")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_rejects_non_s3_bare_literal_to_reader(running_server):
    payload = _post(running_server, "SELECT * FROM read_parquet('http://evil/x')")
    assert "error" in payload
    assert "forbidden" in payload["error"].lower()


def test_query_executes_the_same_text_that_was_validated(running_server):
    """run_query() must validate and execute the SAME stripped string - no
    divergence between what the guard checked and what actually runs. A
    trailing comment (stripped by both) must not change the result."""
    payload = _post(running_server, "SELECT 42 AS x -- trailing comment, stripped either way")
    assert "error" not in payload
    assert payload["rows"] == [[42]]


def test_run_query_self_heals_when_worker_never_stops(monkeypatch):
    """If con.interrupt() doesn't actually stop the worker (e.g. it's stuck
    inside a call with no interrupt point), an uncapped join would hang
    forever holding query_lock and starve every future /query. The bounded
    join must instead hard-exit the process (os._exit) so k8s restarts the
    pod - verified here by faking os._exit to raise instead of actually
    exiting the test process."""
    import service.duckdb_probe_server as svc
    monkeypatch.setattr(svc, "_INTERRUPT_JOIN_TIMEOUT_S", 1)

    class StuckCon:
        def interrupt(self):
            pass  # simulates a worker with no interrupt point

        def sql(self, sql):
            import time
            time.sleep(10)  # never actually stops within the bound
            raise AssertionError("should never return - test would hang otherwise")

    exit_calls = []

    def fake_exit(code):
        exit_calls.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(svc.os, "_exit", fake_exit)

    with pytest.raises(SystemExit):
        svc.run_query(StuckCon(), "SELECT 1", timeout_s=1)
    assert exit_calls == [1]


def test_health_stays_responsive_during_slow_query():
    """ThreadingHTTPServer + a per-connection query_lock: a slow /query must
    not block /health (concurrency), and the query itself must still be the
    only one touching `con` at a time (serialized, not racing)."""
    con = duckdb.connect()
    con.execute("CREATE TABLE slow AS SELECT range AS id FROM range(300000)")
    handler = make_handler(con, timeout_s=10)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        results = {}

        def run_slow_query():
            results["query"] = _post(
                base_url, "SELECT count(*) FROM slow a, slow b LIMIT 1"
            )

        qthread = threading.Thread(target=run_slow_query)
        qthread.start()
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                assert resp.status == 200
        finally:
            qthread.join(timeout=15)
        assert "query" in results
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_build_app_disables_local_filesystem():
    """build_app() (the real server startup path) must disable local
    filesystem access at the engine level - a backstop for the token guard
    above in case a query slips past the regex. Asserted functionally
    (a local read must fail) rather than via current_setting()/
    duckdb_settings(), which don't reliably echo this particular setting's
    value back on the installed duckdb version even though it is enforced."""
    from service.duckdb_probe_server import build_app

    con = build_app()
    with pytest.raises(Exception, match="disabled"):
        con.sql("SELECT * FROM read_text('/etc/hostname')").fetchall()
