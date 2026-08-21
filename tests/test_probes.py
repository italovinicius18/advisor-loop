import duckdb
import pytest
from advisor_loop.probes.duckdb_probe import DuckDBProbe, ProbeTimeout, ProbeQueryError
from advisor_loop.probes.budget import BudgetedProbe, ProbeBudgetExceeded


@pytest.fixture
def store(tmp_path):
    p = tmp_path / "t.parquet"
    duckdb.sql("SELECT range AS id, 'x' AS name FROM range(1000)").write_parquet(str(p))
    return {"store_sales": str(p)}


@pytest.fixture
def multi_table(tmp_path):
    p1 = tmp_path / "store.parquet"
    p2 = tmp_path / "store_returns.parquet"
    duckdb.sql("SELECT range AS id FROM range(100)").write_parquet(str(p1))
    duckdb.sql("SELECT range AS id FROM range(50)").write_parquet(str(p2))
    return {"store": str(p1), "store_returns": str(p2)}


def test_stats_and_sample(store):
    probe = DuckDBProbe(store)
    assert probe.list_tables() == ["store_sales"]
    st = probe.table_stats("store_sales")
    assert st["row_count"] == 1000 and any(c["name"] == "id" for c in st["columns"])
    rows = probe.sample_query("SELECT id FROM store_sales LIMIT 5")
    assert len(rows) == 5


def test_budget_enforced(store):
    b = BudgetedProbe(DuckDBProbe(store), max_probes=2)
    b.table_stats("store_sales")
    with pytest.raises(ValueError):
        b.sample_query("SELECT * FROM store_sales")   # sem LIMIT -> rejeita
    b.sample_query("SELECT * FROM store_sales LIMIT 1")
    with pytest.raises(ProbeBudgetExceeded):
        b.list_tables()


def test_comment_rejection(store):
    """Comments that hide LIMIT should be rejected."""
    b = BudgetedProbe(DuckDBProbe(store), max_probes=8)
    with pytest.raises(ValueError, match="LIMIT explicito"):
        b.sample_query("SELECT * FROM store_sales -- limit 1")
    with pytest.raises(ValueError, match="LIMIT explicito"):
        b.sample_query("SELECT * FROM store_sales /* limit 1 */")


def test_non_select_rejection(store):
    """Non-SELECT/WITH statements should be rejected."""
    b = BudgetedProbe(DuckDBProbe(store), max_probes=8)
    with pytest.raises(ValueError, match="SELECT/WITH"):
        b.sample_query("PRAGMA database_list; -- limit 1")
    with pytest.raises(ValueError, match="SELECT/WITH"):
        b.sample_query("INSERT INTO store_sales VALUES (1, 'x'); -- limit 1")


def test_timeout_enforcement(store):
    """Heavy query should timeout with small timeout_s."""
    probe = DuckDBProbe(store, timeout_s=1)
    with pytest.raises(ProbeTimeout):
        probe.sample_query(
            "SELECT count(*) FROM store_sales a, store_sales b, store_sales c, store_sales d LIMIT 1"
        )


def test_multi_table_substitution(multi_table):
    """Table names should be substituted with word boundaries."""
    probe = DuckDBProbe(multi_table)
    rows = probe.sample_query("SELECT count(*) FROM store_returns LIMIT 1")
    assert len(rows) == 1
    assert rows[0][0] == 50


def test_s3_config_sets_up_secret_without_breaking_local_probes(store):
    """s3_config, when provided, must install httpfs/delta and register a MinIO
    secret without crashing - local (non-s3) tables must still work normally."""
    probe = DuckDBProbe(store, s3_config={
        "endpoint": "localhost:9000", "key_id": "minioadmin", "secret": "minioadmin",
        "use_ssl": False, "url_style": "path",
    })
    assert probe.list_tables() == ["store_sales"]
    st = probe.table_stats("store_sales")
    assert st["row_count"] == 1000


def test_s3_config_none_keeps_local_behavior_identical(store):
    """Default (s3_config=None) must behave exactly as before."""
    probe = DuckDBProbe(store)
    assert probe.list_tables() == ["store_sales"]
    rows = probe.sample_query("SELECT id FROM store_sales LIMIT 3")
    assert len(rows) == 3


def test_available_tables_does_not_spend_budget(store):
    """available_tables() is metadata only (no data scan) - it must not
    consume the probe budget, unlike list_tables()."""
    b = BudgetedProbe(DuckDBProbe(store), max_probes=1)
    assert b.available_tables() == ["store_sales"]
    assert b.used == 0
    b.list_tables()          # the one budgeted call still succeeds
    assert b.used == 1
    with pytest.raises(ProbeBudgetExceeded):
        b.list_tables()


def test_sample_query_type_error_becomes_probe_query_error_not_crash(tmp_path):
    """Untyped-Silver reality: all columns are VARCHAR. sum(str_col) raises a
    DuckDB BinderException - it must surface as ProbeQueryError, not a raw
    duckdb exception (which would crash the campaign uncaught)."""
    p = tmp_path / "t.parquet"
    duckdb.sql("SELECT '1' AS str_col FROM range(10)").write_parquet(str(p))
    probe = DuckDBProbe({"t": str(p)})
    with pytest.raises(ProbeQueryError):
        probe.sample_query("SELECT sum(str_col) FROM t LIMIT 1")
    # a valid aggregation that needs no cast still works
    rows = probe.sample_query("SELECT count(*) FROM t LIMIT 1")
    assert rows[0][0] == 10


def test_table_stats_query_error_becomes_probe_query_error(tmp_path):
    """table_stats against a registered-but-missing file -> genuine DuckDB
    IOException at query time -> must surface as ProbeQueryError, not crash."""
    probe = DuckDBProbe({"ghost": str(tmp_path / "does-not-exist.parquet")})
    with pytest.raises(ProbeQueryError):
        probe.table_stats("ghost")


def test_probe_budget_and_remaining_are_budget_free(store):
    """probe_budget()/remaining() must expose the budget transparently to the
    advisor without themselves costing a probe."""
    b = BudgetedProbe(DuckDBProbe(store), max_probes=3)
    assert b.probe_budget() == 3
    assert b.remaining() == 3
    b.table_stats("store_sales")
    assert b.remaining() == 2
    assert b.probe_budget() == 3          # total budget never changes
    assert b.used == 1                    # neither call spent a probe itself
