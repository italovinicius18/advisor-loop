"""Top-k do oraculo: o gerador de campanhas acha a k-esima CELULA distinta da matriz."""
import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX = str(ROOT / "data/matrix/training_data.parquet")


def _gen():
    spec = importlib.util.spec_from_file_location("gen_sf_campaigns", ROOT / "scripts/gen_sf_campaigns.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shape(c):
    return (c["executor_cores"], c["executor_memory_gb"], c["executor_instances"])


def test_rank1_reproduces_existing_replay_campaign():
    cells = _gen().oracle_cells(MATRIX, 500, rank=1)
    camp = yaml.safe_load(open(ROOT / "campaigns/sf500-replay.yaml"))
    for q, c in camp["configs"].items():
        assert _shape(cells[q]) == _shape(c)


def test_tpch_campaigns_use_tpch_dataset_queries_and_matrix(tmp_path, monkeypatch):
    gen = _gen()
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["gen", "--benchmark", "tpch", "--sfs", "500", "750",
                                     "--image", "img-x", "--matrix",
                                     str(ROOT / "data/matrix/tpch_training_data.parquet")])
    gen.main()
    rep = yaml.safe_load(open(tmp_path / "campaigns/tpch-sf500-replay.yaml"))
    assert rep["datasets"] == ["tpch-sf500"] and rep["queries"] == ["q01", "q09", "q18"]
    assert _shape(rep["configs"]["q09"]) == (12, 48, 6)            # vencedora exp06 sf500 q09
    a2 = yaml.safe_load(open(tmp_path / "campaigns/tpch-sf500-a2-haiku.yaml"))
    assert a2["arm"] == "hybrid" and a2["anchor"] == "llm" and "haiku" in a2["provider"]
    a0 = yaml.safe_load(open(tmp_path / "campaigns/tpch-sf500-a0-heuristic.yaml"))
    assert a0["anchor"] == "heuristic"
    assert (tmp_path / "campaigns/tpch-sf500-random.yaml").exists()
    big = yaml.safe_load(open(tmp_path / "campaigns/tpch-sf750-a2-haiku.yaml"))
    assert big["pinned"]["shuffle_partitions"] == 2000 and big["baseline_config"]["executor_instances"] == 2
    assert not (tmp_path / "campaigns/tpch-sf750-random.yaml").exists()   # sem random em SF>=750
    pod = (tmp_path / "kubernetes/campaign-pods/advisor-controller-tpch-sf500-a0-heuristic.yaml").read_text()
    assert "img-x" in pod and "campaigns/tpch-sf500-a0-heuristic.yaml" in pod


def test_rank2_and_rank3_are_distinct_cells_in_order():
    gen = _gen()
    r1, r2, r3 = (gen.oracle_cells(MATRIX, 500, rank=k) for k in (1, 2, 3))
    # sf500 q23: 12c x 6w x 48g (490.6s) -> 8c x 6w x 28g (591.1s) -> 12c x 4w x 48g (593.5s)
    assert _shape(r1["q23"]) == (12, 48, 6)
    assert _shape(r2["q23"]) == (8, 28, 6)
    assert _shape(r3["q23"]) == (12, 48, 4)
    for q in r1:
        assert len({_shape(r1[q]), _shape(r2[q]), _shape(r3[q])}) == 3
        assert r1[q]["t_star_s"] <= r2[q]["t_star_s"] <= r3[q]["t_star_s"]
