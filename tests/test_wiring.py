import yaml
from advisor_loop.wiring import build_deps
from advisor_loop.backends.replay import ReplayBackend
from advisor_loop.backends.proposal import ProposalBackend


def _campaign(tmp_path, name="c", **over):
    c = {"name": name, "level": "L0", "provider": "m", "datasets": ["sf1"],
         "queries": ["q1"], "objective": "min_time", "tries": 1, "memory": "off", **over}
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(c))
    return p


def _root(tmp_path):
    root = tmp_path / "root"
    (root / "packs").mkdir(parents=True)
    (root / "packs" / "cluster.yaml").write_text(yaml.safe_dump({
        "cluster": {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
                    "max_pods": 20, "total_cores": 100, "total_memory_gb": 400}}))
    (root / "data").mkdir()
    import duckdb
    duckdb.sql("SELECT 1 AS x").write_parquet(str(root / "data" / "reference.parquet"))
    return root


def test_build_deps_defaults_to_replay_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    root = _root(tmp_path)
    camp = _campaign(tmp_path, "replaycamp")
    deps = build_deps(camp, root=root)
    assert isinstance(deps.backend, ReplayBackend)


def test_build_deps_selects_proposal_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    root = _root(tmp_path)
    camp = _campaign(tmp_path, "proposalcamp", backend="proposal")
    deps = build_deps(camp, root=root)
    assert isinstance(deps.backend, ProposalBackend)


def test_build_deps_reads_s3_block_and_wires_probe_factory(tmp_path, monkeypatch):
    """datasets.yaml with a top-level s3: block must be passed through to the
    DuckDBProbe built for a dataset, without crashing probe_factory construction."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    root = _root(tmp_path)
    import duckdb
    tbl = root / "data" / "store_sales.parquet"
    duckdb.sql("SELECT range AS id FROM range(10)").write_parquet(str(tbl))
    (root / "datasets.yaml").write_text(yaml.safe_dump({
        "s3": {"endpoint": "localhost:9000", "key_id": "minioadmin",
               "secret": "minioadmin", "use_ssl": False, "url_style": "path"},
        "sf1": {"store_sales": str(tbl)},
    }))
    camp = _campaign(tmp_path, "s3camp")
    deps = build_deps(camp, root=root)
    from advisor_loop.episode import Episode
    ep = Episode(workload="w", dataset="sf1", queries=["q1"], provider="m")
    probe = deps.probe_factory(ep)
    assert probe is not None
    assert probe.list_tables() == ["store_sales"]


def test_build_deps_probe_service_url_selects_remote_probe(tmp_path, monkeypatch):
    """datasets.yaml with a top-level probe_service.url block must wire the
    remote in-cluster probe instead of the local tunnel-backed DuckDBProbe."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    root = _root(tmp_path)
    import duckdb
    tbl = root / "data" / "store_sales.parquet"
    duckdb.sql("SELECT range AS id FROM range(10)").write_parquet(str(tbl))
    (root / "datasets.yaml").write_text(yaml.safe_dump({
        "probe_service": {"url": "http://localhost:8000"},
        "sf1": {"store_sales": str(tbl)},
    }))
    camp = _campaign(tmp_path, "remotecamp")
    deps = build_deps(camp, root=root)
    from advisor_loop.episode import Episode
    from advisor_loop.probes.remote_probe import RemoteDuckDBProbe
    ep = Episode(workload="w", dataset="sf1", queries=["q1"], provider="m")
    probe = deps.probe_factory(ep)
    assert probe is not None
    assert isinstance(probe._p, RemoteDuckDBProbe)
    assert probe.list_tables() == ["store_sales"]


def test_wiring_selects_spark_k8s(tmp_path, monkeypatch):
    """spark_k8s backend campaigns read a top-level spark: block (ssh_cmd, image)
    from datasets.yaml and wire SparkK8sBackend(SparkSubmitter(...))."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    root = _root(tmp_path)
    (root / "datasets.yaml").write_text(yaml.safe_dump({
        "spark": {"ssh_cmd": ["ssh", "host"], "image": "ghcr.io/x/spark-advisor:t"},
        "sf1": {}}))
    camp = _campaign(tmp_path, "sparkcamp", backend="spark_k8s", mode="optimize")
    deps = build_deps(camp, root=root)
    from advisor_loop.backends.spark_k8s import SparkK8sBackend
    assert isinstance(deps.backend, SparkK8sBackend)


def test_build_deps_without_datasets_yaml_keeps_probe_factory_none(tmp_path, monkeypatch):
    """Backward-compat: no datasets.yaml -> probe_factory returns None (L0 behavior)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    root = _root(tmp_path)
    camp = _campaign(tmp_path, "nodscamp")
    deps = build_deps(camp, root=root)
    from advisor_loop.episode import Episode
    ep = Episode(workload="w", dataset="sf1", queries=["q1"], provider="m")
    assert deps.probe_factory(ep) is None
