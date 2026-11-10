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
    # 3 ranks de replay + 2 ancoras x 2 execucoes por SF
    assert len(names) == 14 and names[0] == "sf100-rep-replay"
    assert "sf500-rep-a2-haiku-r2" in names and "sf100-rep-a0-heuristic-r2" in names
    r2 = yaml.safe_load(open(tmp_path / "campaigns/sf100-rep-a0-heuristic-r2.yaml"))
    assert r2["name"] == "sf100-rep-a0-heuristic-r2" and r2["anchor"] == "heuristic"
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
    # so dispara com o cluster ocioso (fora a infraestrutura) por 30 min seguidos
    assert "wait_cluster_idle" in q and q.index("wait_cluster_idle\n") < q.index("for name in")
    assert "minio-" in q and "duckdb-probe-" in q and "IDLE_NEEDED=15" in q


def test_queue_idle_wait_logic(tmp_path):
    """Executa so a funcao de espera do script com um kubectl falso."""
    import subprocess
    g = _gen()
    head = g.QUEUE_HEAD
    fake = tmp_path / "kubectl"
    state = tmp_path / "n"
    state.write_text("0")
    # 3 primeiras consultas: cluster ocupado; depois so infraestrutura
    fake.write_text("""#!/usr/bin/env bash
n=$(cat %s); echo $((n+1)) > %s
if [ "$n" -lt 3 ]; then echo "exp05-x Running"; fi
echo "minio-0 Running"; echo "duckdb-probe-abc Running"; echo "prometheus-1 Pending"; echo "advisor-queue Running"
""" % (state, state))
    fake.chmod(0o755)
    script = head.replace("IDLE_NEEDED=15", "IDLE_NEEDED=2").replace("sleep 120", "sleep 0") + "wait_cluster_idle\necho PRONTO\n"
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         env={"PATH": f"{tmp_path}:/usr/bin:/bin"})
    assert "PRONTO" in out.stdout
    assert int(state.read_text()) == 5        # 3 ocupadas + 2 ociosas seguidas
