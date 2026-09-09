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
    # JARs assados na imagem: nunca resolver via Ivy em runtime
    assert "spark.jars.packages" not in args
    # deploy-mode cluster: env precisa chegar ao driver via driverEnv.*
    assert "spark.kubernetes.driverEnv.SF=sf1" in args
    assert "spark.kubernetes.driverEnv.QUERY=q03" in args
    assert "spark.kubernetes.driverEnv.RESULT_S3=s3://advisor-results/r123.json" in args
    assert "spark.kubernetes.driverEnv.MINIO_ENDPOINT=http://minio:9000" in args
    # driver pod name previsivel p/ poll/logs/cleanup
    assert "spark.kubernetes.driver.pod.name=advisor-r123-driver" in args
    env = {e["name"]: e["value"] for e in spec["containers"][0]["env"]}
    assert env["SF"] == "sf1" and env["QUERY"] == "q03" and env["RESULT_S3"].endswith("r123.json")
    # sem node_selector: nenhuma restricao de no
    assert "nodeSelector" not in spec
    assert "node.selector" not in args


def test_node_selector_pins_launcher_driver_executors():
    m = render_launcher(_cv(), sf="sf1", query="q03", run_id="r9",
                        image="img", result_s3="s3://advisor-results/r9.json",
                        node_selector={"kubernetes.io/hostname": "ids-pb"})
    doc = yaml.safe_load(m)
    assert doc["spec"]["nodeSelector"] == {"kubernetes.io/hostname": "ids-pb"}
    args = " ".join(doc["spec"]["containers"][0]["args"])
    assert "spark.kubernetes.driver.node.selector.kubernetes.io/hostname=ids-pb" in args
    assert "spark.kubernetes.executor.node.selector.kubernetes.io/hostname=ids-pb" in args


def test_pod_template_affinity_targets_egress_pool():
    # via multi-no: podTemplateFile p/ driver+executors + nodeAffinity In [...]
    # no pod launcher. Necessario p/ configs que nao cabem num no so (q23/q50).
    m = render_launcher(_cv(), sf="sf100", query="q23", run_id="r7",
                        image="img", result_s3="s3://advisor-results/r7.json",
                        pod_template_file="/opt/advisor/pod-template-egress.yaml",
                        node_affinity_hostnames=["ids-es", "ids-rn", "ids-pb"])
    doc = yaml.safe_load(m)
    args = " ".join(doc["spec"]["containers"][0]["args"])
    assert "spark.kubernetes.driver.podTemplateFile=/opt/advisor/pod-template-egress.yaml" in args
    assert "spark.kubernetes.executor.podTemplateFile=/opt/advisor/pod-template-egress.yaml" in args
    # sem node.selector de igualdade nesta via
    assert "node.selector" not in args
    # o pod launcher usa nodeAffinity (nao nodeSelector) ao mesmo pool
    assert "nodeSelector" not in doc["spec"]
    terms = doc["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    expr = terms[0]["matchExpressions"][0]
    assert expr["key"] == "kubernetes.io/hostname"
    assert expr["operator"] == "In"
    assert expr["values"] == ["ids-es", "ids-rn", "ids-pb"]
