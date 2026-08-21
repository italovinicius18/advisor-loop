import json
import threading
import urllib.request
from http.server import HTTPServer

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
    server = HTTPServer(("127.0.0.1", 0), handler)
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
    server = HTTPServer(("127.0.0.1", 0), handler)
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
