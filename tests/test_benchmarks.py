"""Dataset -> (benchmark, sf): TPC-DS mantem os nomes legados 'sf<N>'; TPC-H usa
'tpch-sf<N>' (mesmo SF, outras tabelas e queries)."""
import pytest

from advisor_loop.benchmarks import benchmark_of, queries_subdir, sf_of, sf_segment


def test_legacy_tpcds_names_unchanged():
    assert benchmark_of("sf500") == "tpcds" and sf_of("sf500") == 500
    assert sf_segment("sf500") == "sf500"
    assert queries_subdir("sf500") == "queries"


def test_tpch_prefix():
    assert benchmark_of("tpch-sf750") == "tpch" and sf_of("tpch-sf750") == 750
    assert sf_segment("tpch-sf750") == "sf750"
    assert queries_subdir("tpch-sf750") == "queries-tpch"


def test_unknown_benchmark_rejected():
    with pytest.raises(ValueError):
        benchmark_of("job-sf10")


def test_report_parses_tpch_dataset():
    from analysis.report import _sf_of
    assert _sf_of("tpch-sf500") == 500 and _sf_of("sf100") == 100


def test_driver_resolution_matches_controller():
    # o driver e um arquivo solto na imagem (nao importa advisor_loop): a
    # resolucao dele tem que bater com a do controller.
    from spark_advisor.driver import resolve_dataset
    b = resolve_dataset("tpch-sf500")
    assert b["silver"] == "s3a://silver/sf500/tpch/delta/parquet"
    assert b["queries_dir"] == "/opt/advisor/queries-tpch"
    assert b["tables"] == ["customer", "lineitem", "nation", "orders", "part",
                           "partsupp", "region", "supplier"]
    d = resolve_dataset("sf500")
    assert d["silver"] == "s3a://silver/sf500/delta/parquet"
    assert d["queries_dir"] == "/opt/advisor/queries" and len(d["tables"]) == 24
