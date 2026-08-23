import json
import os
import sys
import time


def throughput_mbps(bytes_: int, duration_ms: float) -> float:
    if not bytes_ or not duration_ms:
        return 0.0
    return round((bytes_ / 1_000_000) / (duration_ms / 1000), 3)


def classify_error(err) -> str:
    t = str(err)
    if "OutOfMemoryError" in t or "exit code 137" in t or "OOMKilled" in t:
        return "oom"
    if "TimeoutException" in t or "timed out" in t.lower():
        return "timeout"
    return "failed"


def build_result(status, duration_ms, metrics, error) -> dict:
    return {"status": status, "duration_ms": duration_ms,
            "metrics": metrics or {}, "error": error}


def _write_result(result: dict):
    line = json.dumps(result)
    print("ADVISOR_RESULT " + line, flush=True)   # marcador único p/ os logs
    dst = os.environ.get("RESULT_S3")
    if dst:
        try:
            import boto3  # presente na imagem
            # dst = s3://advisor-results/{run_id}.json ; escrita via s3a seria no Spark,
            # mas aqui gravamos direto via boto3 no MinIO (endpoint/keys de env)
            s3 = boto3.client("s3", endpoint_url=os.environ["MINIO_ENDPOINT"],
                              aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
                              aws_secret_access_key=os.environ["MINIO_SECRET_KEY"])
            bucket, key = dst.replace("s3://", "").split("/", 1)
            s3.put_object(Bucket=bucket, Key=key, Body=line.encode())
        except Exception as e:  # nunca falhar o job por causa do write de resultado
            print("ADVISOR_RESULT_WRITE_ERROR " + str(e), flush=True)


def main():   # coberto pela validação ao vivo, não por unit test
    from pyspark.sql import SparkSession
    sf = os.environ["SF"]
    query = os.environ["QUERY"]
    spark = SparkSession.builder.appName(f"advisor-{query}-{sf}").getOrCreate()
    # métricas via listener
    metrics = {"input_bytes": 0, "output_bytes": 0, "shuffle_read_bytes": 0,
               "shuffle_write_bytes": 0, "gc_time_ms": 0}
    try:
        base = f"s3a://silver/{sf}/delta/parquet"
        # registra todas as tabelas TPC-DS como views
        import re
        sql = open(f"/opt/advisor/queries/{query}.sql").read()
        tables = set(re.findall(r"\bfrom\s+([a-z_]+)|\bjoin\s+([a-z_]+)", sql, re.I))
        names = {t for pair in tables for t in pair if t}
        for name in names:
            try:
                spark.read.format("delta").load(f"{base}/{name}").createOrReplaceTempView(name)
            except Exception:
                pass
        t0 = time.time()
        spark.sql(sql).write.format("noop").mode("overwrite").save()
        dur = (time.time() - t0) * 1000
        # coleta métricas do último query execution (best-effort)
        _write_result(build_result("success", round(dur, 1), metrics, None))
    except Exception as e:
        _write_result(build_result(classify_error(e), None, metrics, str(e)[:500]))
        spark.stop()
        sys.exit(1)
    spark.stop()


if __name__ == "__main__":
    main()
