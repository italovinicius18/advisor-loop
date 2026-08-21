import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import duckdb
import pytest

from service.duckdb_probe_server import make_handler
from advisor_loop.probes.budget import BudgetedProbe
from advisor_loop.probes.duckdb_probe import ProbeQueryError
from advisor_loop.probes.remote_probe import RemoteDuckDBProbe


class _FakeS3Handler(BaseHTTPRequestHandler):
    """Minimal Range-GET-capable static file server standing in for MinIO/S3
    in tests. The server-side guard (service/duckdb_probe_server.py) now
    requires every S3-reader call to use a bare 's3://...' literal - so
    these tests need real read_parquet('s3://...') execution, not a local
    path, to stay a genuine end-to-end check. `root` is set per subclass
    (see _make_fake_s3_handler) to the directory serving as the fake bucket
    root; DuckDB's httpfs issues Range GETs for the parquet footer/row
    groups, which stdlib http.server doesn't support out of the box."""
    root: str = ""

    def log_message(self, fmt, *args):
        pass

    def do_HEAD(self):
        self._serve(head_only=True)

    def do_GET(self):
        self._serve(head_only=False)

    def _serve(self, head_only: bool) -> None:
        local = os.path.join(self.root, self.path.lstrip("/"))
        if not os.path.isfile(local):
            self.send_response(404)
            self.end_headers()
            return
        size = os.path.getsize(local)
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d+)-(\d*)", rng)
            start, end = int(m.group(1)), (int(m.group(2)) if m.group(2) else size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
        else:
            start, length = 0, size
            self.send_response(200)
            self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if not head_only:
            with open(local, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))


def _make_fake_s3_handler(root):
    return type("FakeS3Handler", (_FakeS3Handler,), {"root": str(root)})


@pytest.fixture
def server_and_tables(tmp_path):
    """Real local http.server driving a real duckdb connection, plus a
    second local http.server standing in for MinIO/S3 (see _FakeS3Handler) -
    no real MinIO, but a genuine end-to-end read_parquet('s3://...') against
    real bytes, matching what the server-side guard now requires."""
    bucket_dir = tmp_path / "test-bucket"
    bucket_dir.mkdir()
    p = bucket_dir / "store_sales.parquet"
    duckdb.sql("SELECT range AS id, 'x' AS name FROM range(1000)").write_parquet(str(p))
    tables = {"store_sales": "s3://test-bucket/store_sales.parquet"}

    s3_server = HTTPServer(("127.0.0.1", 0), _make_fake_s3_handler(tmp_path))
    s3_thread = threading.Thread(target=s3_server.serve_forever, daemon=True)
    s3_thread.start()

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs")
    con.execute(
        "CREATE OR REPLACE SECRET minio (TYPE S3, KEY_ID 'x', SECRET 'x', "
        f"ENDPOINT '127.0.0.1:{s3_server.server_port}', URL_STYLE 'path', USE_SSL false)"
    )
    handler = make_handler(con, timeout_s=10)
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", tables
    finally:
        server.shutdown()
        thread.join(timeout=2)
        s3_server.shutdown()
        s3_thread.join(timeout=2)


def test_list_tables_is_client_side(server_and_tables):
    base_url, tables = server_and_tables
    probe = RemoteDuckDBProbe("http://127.0.0.1:1", tables)  # unreachable port
    assert probe.list_tables() == ["store_sales"]


def test_table_stats(server_and_tables):
    base_url, tables = server_and_tables
    probe = RemoteDuckDBProbe(base_url, tables)
    st = probe.table_stats("store_sales")
    assert st["row_count"] == 1000
    assert any(c["name"] == "id" for c in st["columns"])


def test_sample_query(server_and_tables):
    base_url, tables = server_and_tables
    probe = RemoteDuckDBProbe(base_url, tables)
    rows = probe.sample_query("SELECT id FROM store_sales ORDER BY id LIMIT 5")
    assert rows == [(0,), (1,), (2,), (3,), (4,)]


def test_bad_sql_becomes_probe_query_error(server_and_tables):
    base_url, tables = server_and_tables
    probe = RemoteDuckDBProbe(base_url, tables)
    with pytest.raises(ProbeQueryError):
        probe.sample_query("SELECT sum(name) FROM store_sales LIMIT 1")


def test_non_select_becomes_probe_query_error(server_and_tables):
    base_url, tables = server_and_tables
    probe = RemoteDuckDBProbe(base_url, tables)
    with pytest.raises(ProbeQueryError):
        probe.sample_query("PRAGMA database_list")


def test_table_stats_missing_table_raises_keyerror(server_and_tables):
    base_url, tables = server_and_tables
    probe = RemoteDuckDBProbe(base_url, tables)
    with pytest.raises(KeyError):
        probe.table_stats("ghost")


def test_budgeted_probe_keeps_remote_probes_own_timeout_default(server_and_tables):
    """BudgetedProbe(RemoteDuckDBProbe(...)) with no explicit timeout_s must
    not clobber RemoteDuckDBProbe's own 35s default down to some other
    value - it comfortably exceeds the server's QUERY_TIMEOUT_S=30s."""
    base_url, tables = server_and_tables
    probe = RemoteDuckDBProbe(base_url, tables)
    assert probe.timeout_s == 35
    BudgetedProbe(probe)
    assert probe.timeout_s == 35


def test_network_error_becomes_probe_query_error(tmp_path):
    p = tmp_path / "t.parquet"
    duckdb.sql("SELECT 1 AS id").write_parquet(str(p))
    probe = RemoteDuckDBProbe("http://127.0.0.1:1", {"t": str(p)}, timeout_s=1)
    with pytest.raises(ProbeQueryError):
        probe.table_stats("t")
