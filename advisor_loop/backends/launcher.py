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
                     image: str, result_s3: str) -> str:
    args = FIXED_CONFS + [
        "--conf", f"spark.kubernetes.container.image={image}",
        "--conf", "spark.hadoop.fs.s3a.endpoint=http://minio:9000",
        "--conf", "spark.hadoop.fs.s3a.access.key=minioadmin",
        "--conf", "spark.hadoop.fs.s3a.secret.key=minioadmin",
        "--conf", "spark.hadoop.fs.s3a.path.style.access=true",
        "--conf", "spark.hadoop.fs.s3a.connection.ssl.enabled=false",
        "--conf", "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension",
        "--conf", "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog",
        "--conf", "spark.jars.packages=io.delta:delta-spark_2.12:3.2.0",
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
