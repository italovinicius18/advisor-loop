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
                     image: str, result_s3: str, node_selector: dict | None = None) -> str:
    driver_pod = f"advisor-{run_id}-driver"
    # targeting de no opcional: clusters heterogeneos (DNS/egress irregular, imagem
    # cacheada so em alguns nos) precisam fixar driver+executors num no conhecido.
    node_confs = []
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
            **({"nodeSelector": node_selector} if node_selector else {}),
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
