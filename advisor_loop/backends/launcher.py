import yaml
from advisor_loop.backends.spark_conf import spark_confs
from advisor_loop.episode import ConfigVector

FIXED_CONFS = [
    "--master", "k8s://https://kubernetes.default.svc:443",
    "--deploy-mode", "cluster",
    "--conf", "spark.kubernetes.namespace=italo-unb",
    "--conf", "spark.kubernetes.authenticate.driver.serviceAccountName=ns-admin",
]

def render_launcher(config: ConfigVector, sf: str, query: str, run_id: str,
                     image: str, result_s3: str, node_selector: dict | None = None,
                     pod_template_file: str | None = None,
                     node_affinity_hostnames: list[str] | None = None,
                     tries: int = 1, warmup: int = 1) -> str:
    driver_pod = f"advisor-{run_id}-driver"
    # targeting de no opcional. Duas vias mutuamente exclusivas:
    #  - node_selector (igualdade): fixa driver+executors num UNICO no conhecido
    #    (usado no smoke local; simples, mas nao mira um conjunto de nos).
    #  - pod_template_file: nodeAffinity In [...] assada na imagem -> mira o POOL
    #    de egress (>= 2 nos), unica via sob RBAC namespace-scoped (sem rotular
    #    nodes). Necessario p/ configs que nao cabem num no so (q23/q50 = 72c).
    node_confs = []
    if pod_template_file:
        node_confs += ["--conf", f"spark.kubernetes.driver.podTemplateFile={pod_template_file}"]
        node_confs += ["--conf", f"spark.kubernetes.executor.podTemplateFile={pod_template_file}"]
    for k, v in (node_selector or {}).items():
        node_confs += ["--conf", f"spark.kubernetes.driver.node.selector.{k}={v}"]
        node_confs += ["--conf", f"spark.kubernetes.executor.node.selector.{k}={v}"]
    args = FIXED_CONFS + node_confs + [
        "--conf", f"spark.kubernetes.container.image={image}",
        # IfNotPresent: reusa a imagem cacheada no no; Always forcaria um HEAD ao
        # ghcr.io e o DNS de alguns nos do cluster (ex.: ids-pb) e intermitente.
        "--conf", "spark.kubernetes.container.image.pullPolicy=IfNotPresent",
        # nome do driver pod previsivel: usado para poll/logs/cleanup no submitter
        "--conf", f"spark.kubernetes.driver.pod.name={driver_pod}",
        # em deploy-mode cluster o driver roda em pod separado; o bloco env do
        # pod launcher NAO chega ao driver -> propagar via driverEnv.*
        "--conf", f"spark.kubernetes.driverEnv.SF={sf}",
        "--conf", f"spark.kubernetes.driverEnv.QUERY={query}",
        "--conf", f"spark.kubernetes.driverEnv.RUN_ID={run_id}",
        # tries em sessao + warm-up: protocolo quente (ver driver.py)
        "--conf", f"spark.kubernetes.driverEnv.TRIES={tries}",
        "--conf", f"spark.kubernetes.driverEnv.WARMUP={warmup}",
        "--conf", f"spark.kubernetes.driverEnv.RESULT_S3={result_s3}",
        "--conf", "spark.kubernetes.driverEnv.MINIO_ENDPOINT=http://minio:9000",
        "--conf", "spark.kubernetes.driverEnv.MINIO_ACCESS_KEY=minioadmin",
        "--conf", "spark.kubernetes.driverEnv.MINIO_SECRET_KEY=minioadmin",
        "--conf", "spark.hadoop.fs.s3a.endpoint=http://minio:9000",
        # Hadoop 3.5 usa AWS SDK v2, que exige uma regiao mesmo p/ MinIO;
        # sem isso toda operacao s3a falha ao resolver regiao. us-east-1 e o dummy.
        "--conf", "spark.hadoop.fs.s3a.endpoint.region=us-east-1",
        # Hadoop 3.5 traz o "S3A Analytics Accelerator" (jar extra ausente);
        # forcar o stream classico evita o NoClassDefFoundError e basta p/ MinIO.
        "--conf", "spark.hadoop.fs.s3a.input.stream.type=classic",
        "--conf", "spark.hadoop.fs.s3a.access.key=minioadmin",
        "--conf", "spark.hadoop.fs.s3a.secret.key=minioadmin",
        "--conf", "spark.hadoop.fs.s3a.path.style.access=true",
        "--conf", "spark.hadoop.fs.s3a.connection.ssl.enabled=false",
        "--conf", "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension",
        "--conf", "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog",
        # Spark 4.x liga ANSI por padrao; a Silver e toda-VARCHAR (escrita em
        # Spark 3.5 nao-ANSI), entao sum()/casts implicitos quebrariam. Desligar
        # ANSI mantem a semantica de origem e isola o estudo de config.
        "--conf", "spark.sql.ansi.enabled=false",
        # JARs Delta+S3A estao assados na imagem (docker/spark-advisor/Dockerfile);
        # NAO usar spark.jars.packages -> resolucao Ivy falha no pod (cache/egress).
    ] + spark_confs(config) + [
        "local:///opt/advisor/driver.py",
    ]

    # O POD launcher tambem precisa cair no pool de egress (roda spark-submit da
    # mesma imagem e fica de pe durante a run). Espelha o node_affinity_hostnames
    # do template assado -- o launcher nao le o arquivo da imagem em tempo de
    # render, entao a lista vem via config (unica fonte dinamica p/ este pod).
    pod_targeting: dict = {}
    if node_affinity_hostnames:
        pod_targeting["affinity"] = {"nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
                {"matchExpressions": [{"key": "kubernetes.io/hostname",
                                       "operator": "In",
                                       "values": list(node_affinity_hostnames)}]}]}}}
    elif node_selector:
        pod_targeting["nodeSelector"] = node_selector

    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"advisor-{run_id}",
            "namespace": "italo-unb",
        },
        "spec": {
            "serviceAccountName": "ns-admin",
            "restartPolicy": "Never",
            **pod_targeting,
            "containers": [
                {
                    "name": "spark-submit",
                    "image": image,
                    "command": ["/opt/spark/bin/spark-submit"],
                    "args": args,
                    "env": [
                        {"name": "SF", "value": sf},
                        {"name": "QUERY", "value": query},
                        {"name": "RUN_ID", "value": run_id},
                        {"name": "TRIES", "value": str(tries)},
                        {"name": "WARMUP", "value": str(warmup)},
                        {"name": "RESULT_S3", "value": result_s3},
                        {"name": "MINIO_ENDPOINT", "value": "http://minio:9000"},
                        {"name": "MINIO_ACCESS_KEY", "value": "minioadmin"},
                        {"name": "MINIO_SECRET_KEY", "value": "minioadmin"},
                    ],
                }
            ],
        },
    }
    return yaml.safe_dump(manifest)
