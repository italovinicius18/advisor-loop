import yaml
from advisor_loop.backends.launcher import render_launcher
from tests.test_spark_conf import _cv

def test_render_valid_yaml_with_confs():
    m = render_launcher(_cv(executor_instances=8), sf="sf1", query="q03",
                        run_id="r123", image="ghcr.io/x/spark-advisor:t", result_s3="s3://advisor-results/r123.json")
    doc = yaml.safe_load(m)
    assert doc["kind"] == "Pod"
    assert doc["metadata"]["name"] == "advisor-r123"
    assert doc["metadata"]["namespace"] == "italo-unb"
    spec = doc["spec"]
    assert spec["serviceAccountName"] == "ns-admin"
    assert spec["restartPolicy"] == "Never"
    args = " ".join(spec["containers"][0]["args"])
    assert "spark.executor.instances=8" in args
    assert "k8s://https://kubernetes.default.svc:443" in args
    assert "spark.kubernetes.namespace=italo-unb" in args
    assert "io.delta.sql.DeltaSparkSessionExtension" in args
    env = {e["name"]: e["value"] for e in spec["containers"][0]["env"]}
    assert env["SF"] == "sf1" and env["QUERY"] == "q03" and env["RESULT_S3"].endswith("r123.json")
