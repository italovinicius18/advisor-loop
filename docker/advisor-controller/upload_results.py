"""Sobe os artefatos da campanha (journal jsonl, memory, run.log) para o MinIO,
para que sobrevivam ao fim do pod controller e ao PC do usuario desligado.

Destino: s3://advisor-results/<prefix>/... (prefix = nome da campanha).
Endpoint interno do cluster; credenciais minioadmin. Idempotente; cria o bucket
se faltar. Nunca aborta o pod: erros viram warning no stdout.
"""
import glob
import os
import sys

import boto3
from botocore.client import Config

# ids-pb nao resolve *.cluster.local (usa DNS externo p/ egress). O ClusterIP do
# svc minio e roteavel por IP a partir de ids-pb, entao o default e o IP fixo.
ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://10.50.217.20:9000")
BUCKET = os.environ.get("RESULT_BUCKET", "advisor-results")
PREFIX = os.environ.get("RESULT_PREFIX", "smoke-sf100")
ROOT = "/opt/advisor-loop"


def main() -> int:
    s3 = boto3.client("s3", endpoint_url=ENDPOINT,
                      aws_access_key_id=os.environ.get("MINIO_KEY", "minioadmin"),
                      aws_secret_access_key=os.environ.get("MINIO_SECRET", "minioadmin"),
                      config=Config(s3={"addressing_style": "path"}))
    try:
        s3.head_bucket(Bucket=BUCKET)
    except Exception:
        try:
            s3.create_bucket(Bucket=BUCKET)
        except Exception as e:
            print(f"[upload] aviso: nao criou bucket {BUCKET}: {e}")

    paths = glob.glob(f"{ROOT}/evidence/**/*", recursive=True) + [f"{ROOT}/run.log"]
    n = 0
    for p in paths:
        if not os.path.isfile(p):
            continue
        rel = os.path.relpath(p, ROOT)              # ex.: evidence/cluster-smoke-sf100/journal.jsonl
        key = f"{PREFIX}/{rel}"
        try:
            s3.upload_file(p, BUCKET, key)
            print(f"[upload] s3://{BUCKET}/{key}")
            n += 1
        except Exception as e:
            print(f"[upload] aviso: falhou {p}: {e}")
    print(f"[upload] {n} arquivo(s) enviados para s3://{BUCKET}/{PREFIX}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
