from pathlib import Path
from advisor_loop.knowledge.pack import KnowledgePack
from advisor_loop.episode import Episode

def _write_packs(d: Path):
    (d / "engine.yaml").write_text(
        "engine: spark-3.5\nknobs:\n  shuffle_partitions: 'particoes de shuffle; sobe com volume'\n")
    (d / "cluster.yaml").write_text(
        "cluster:\n  max_cores_per_container: 24\n  max_memory_gb_per_container: 96\n"
        "  max_pods: 20\n  total_cores: 100\n  total_memory_gb: 400\n")

def test_load_render_hash(tmp_path):
    _write_packs(tmp_path)
    pack = KnowledgePack.load(tmp_path)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    ctx = pack.render(ep)
    assert "shuffle_partitions" in ctx and "sf100" in ctx
    assert len(pack.hash) == 16
    assert pack.cluster["max_pods"] == 20
