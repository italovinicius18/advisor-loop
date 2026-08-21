import threading
from http.server import HTTPServer

import duckdb
import pytest

from service.duckdb_probe_server import make_handler
from advisor_loop.probes.duckdb_probe import ProbeQueryError
from advisor_loop.probes.remote_probe import RemoteDuckDBProbe


@pytest.fixture
def server_and_tables(tmp_path):
    """Real local http.server driving a local duckdb connection over a local
    parquet fixture - no MinIO, mirrors the in-cluster deployment shape."""
    p = tmp_path / "store_sales.parquet"
    duckdb.sql("SELECT range AS id, 'x' AS name FROM range(1000)").write_parquet(str(p))
    tables = {"store_sales": str(p)}

    con = duckdb.connect()
    handler = make_handler(con, timeout_s=10)
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", tables
    finally:
        server.shutdown()
        thread.join(timeout=2)


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


def test_network_error_becomes_probe_query_error(tmp_path):
    p = tmp_path / "t.parquet"
    duckdb.sql("SELECT 1 AS id").write_parquet(str(p))
    probe = RemoteDuckDBProbe("http://127.0.0.1:1", {"t": str(p)}, timeout_s=1)
    with pytest.raises(ProbeQueryError):
        probe.table_stats("t")
