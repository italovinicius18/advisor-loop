"""Campanhas das queries representativas por categoria (A scan, B shuffle, C subquery)."""
import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _gen():
    spec = importlib.util.spec_from_file_location("gen_rep", ROOT / "scripts/gen_rep_campaigns.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_rep_campaigns(tmp_path, monkeypatch):
    g = _gen()
    monkeypatch.setattr(g, "ROOT", tmp_path)
    names = g.generate(sfs=[100, 500], image="img-y")
    assert g.QUERIES == ["q19", "q71", "q94", "q66", "q05", "q02"]
    # 3 ranks de replay + 2 ancoras por SF
    assert len(names) == 10 and names[0] == "sf100-rep-replay"
    rep = yaml.safe_load(open(tmp_path / "campaigns/sf500-rep-replay-r2.yaml"))
    assert rep["datasets"] == ["sf500"] and rep["queries"] == g.QUERIES
    assert rep["arm"] == "fixed" and rep["baseline"] is False and set(rep["configs"]) == set(g.QUERIES)
    a0 = yaml.safe_load(open(tmp_path / "campaigns/sf100-rep-a0-heuristic.yaml"))
    assert a0["anchor"] == "heuristic" and a0["knobs"] == "matrix3" and "envelope" not in a0
    a2 = yaml.safe_load(open(tmp_path / "campaigns/sf500-rep-a2-haiku.yaml"))
    assert a2["provider"] == "anthropic/claude-haiku-4.5" and a2["max_iters"] == 8 and a2["patience"] == 3
    pod = (tmp_path / "kubernetes/campaign-pods/advisor-controller-sf500-rep-a2-haiku.yaml").read_text()
    assert "img-y" in pod and "campaigns/sf500-rep-a2-haiku.yaml" in pod
    q = (tmp_path / "kubernetes/campaign-pods/queue_rep.sh").read_text()
    assert " ".join(names) in q
