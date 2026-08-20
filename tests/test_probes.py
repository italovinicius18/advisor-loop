import duckdb
import pytest
from advisor_loop.probes.duckdb_probe import DuckDBProbe
from advisor_loop.probes.budget import BudgetedProbe, ProbeBudgetExceeded


@pytest.fixture
def store(tmp_path):
    p = tmp_path / "t.parquet"
    duckdb.sql("SELECT range AS id, 'x' AS name FROM range(1000)").write_parquet(str(p))
    return {"store_sales": str(p)}


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
