import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_smoke_campaign():
    c = yaml.safe_load((ROOT / "campaigns/cluster-smoke-sf100.yaml").read_text())
    assert c["mode"] == "optimize" and c["backend"] == "spark_k8s"
    assert c["arm"] == "llm" and c["tries"] == 1 and c["level"] == "L0"
    assert c["datasets"] == ["sf100"] and set(c["queries"]) == {"q03", "q23", "q50"}


def test_campaign():
    c = yaml.safe_load((ROOT / "campaigns/cluster-sf100.yaml").read_text())
    assert c["tries"] == 3 and c["max_iters"] == 8 and c["arm"] in ("llm", "random")
    assert c["level"] == "L0" and c["backend"] == "spark_k8s"


def test_datasets_has_sf100():
    ds = yaml.safe_load((ROOT / "datasets.yaml").read_text())
    assert "sf100" in ds
    assert ds["sf100"]["store_sales"] == "s3://silver/sf100/delta/parquet/store_sales"
    assert "spark" in ds and ds["spark"]["ssh_cmd"][0] == "ssh"


def test_matrix_data_present():
    p = ROOT / "data/matrix/training_data.parquet"
    assert p.exists() and p.stat().st_size > 0
    assert (ROOT / "data/matrix/README.md").exists()
