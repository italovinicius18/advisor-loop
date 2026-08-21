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
