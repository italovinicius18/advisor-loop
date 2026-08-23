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


# métricas agregadas por stage que o SparkListener coleta (e que alimentam o
# histórico visto pelo LLM). Puras/testáveis; o listener em si é coberto ao vivo.
_METRIC_KEYS = ("input_bytes", "output_bytes", "shuffle_read_bytes",
                "shuffle_write_bytes", "gc_time_ms", "memory_spill_bytes",
                "disk_spill_bytes", "records_read")


def zero_metrics() -> dict:
    return {k: 0 for k in _METRIC_KEYS}


def accumulate(acc: dict, stage: dict) -> dict:
    """Soma as métricas de um stage no acumulador (chamado pelo listener por
    stage). Ignora chaves ausentes e valores None. Puro → unit-testável."""
    for k in _METRIC_KEYS:
        acc[k] = int(acc.get(k, 0)) + int(stage.get(k, 0) or 0)
    return acc


def _register_metrics_listener(spark, acc):
    """Registra um SparkListener via py4j que acumula taskMetrics por stage em
    `acc`. Degradação graciosa: se o callback server / addSparkListener falhar,
    o job segue com métricas zeradas (nunca derruba a execução)."""
    try:
        from py4j.java_gateway import CallbackServerParameters
        gw = spark.sparkContext._gateway
        # o Java precisa chamar de volta o Python. start_callback_server sobe o
        # servidor numa porta (auto); MAS o Java mantém um CallbackClient apontando
        # p/ a porta default (25334) -> "Connection refused". Fix conhecido:
        # resetar o CallbackClient do Java para a porta REAL do callback server.
        if gw.get_callback_server() is None:
            gw.start_callback_server(CallbackServerParameters(port=0, daemonize=True,
                                                              daemonize_connections=True))
        cb = gw.get_callback_server()
        port = cb.get_listening_port()
        addr = gw.java_gateway_server.getCallbackClient().getAddress()
        gw.java_gateway_server.resetCallbackClient(addr, port)

        class _Listener:
            def onStageCompleted(self, stageCompleted):
                try:
                    tm = stageCompleted.stageInfo().taskMetrics()
                    accumulate(acc, {
                        "input_bytes": tm.inputMetrics().bytesRead(),
                        "records_read": tm.inputMetrics().recordsRead(),
                        "output_bytes": tm.outputMetrics().bytesWritten(),
                        "shuffle_read_bytes": tm.shuffleReadMetrics().totalBytesRead(),
                        "shuffle_write_bytes": tm.shuffleWriteMetrics().bytesWritten(),
                        "gc_time_ms": tm.jvmGCTime(),
                        "memory_spill_bytes": tm.memoryBytesSpilled(),
                        "disk_spill_bytes": tm.diskBytesSpilled(),
                    })
                except Exception as ex:
                    print("ADVISOR_LISTENER_STAGE_ERROR " + str(ex)[:200], flush=True)

            def __getattr__(self, name):
                # o SparkListenerInterface tem ~30 métodos onXxx; os que não
                # tratamos viram no-op (robusto a diferenças entre versões).
                if name.startswith("on"):
                    return lambda *a, **k: None
                raise AttributeError(name)

            class Java:
                implements = ["org.apache.spark.scheduler.SparkListenerInterface"]

        spark.sparkContext._jsc.sc().addSparkListener(_Listener())
        return True
    except Exception as ex:
        print("ADVISOR_LISTENER_INIT_ERROR " + str(ex)[:200], flush=True)
        return False


def _flush_listener_bus(spark, timeout_ms=10000):
    """Espera o listener bus drenar para que os eventos de stage do query já
    tenham sido contabilizados antes de lermos as métricas."""
    try:
        spark.sparkContext._jsc.sc().listenerBus().waitUntilEmpty(timeout_ms)
    except Exception:
        pass


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
    # métricas reais coletadas por um SparkListener (acumulador mutado nos callbacks)
    metrics = zero_metrics()
    _register_metrics_listener(spark, metrics)
    # todas as tabelas TPC-DS (registrar as que existirem na Silver evita
    # a fragilidade de extrair nomes do SQL — joins por virgula/alias)
    TPCDS_TABLES = [
        "call_center", "catalog_page", "catalog_returns", "catalog_sales",
        "customer", "customer_address", "customer_demographics", "date_dim",
        "household_demographics", "income_band", "inventory", "item",
        "promotion", "reason", "ship_mode", "store", "store_returns",
        "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
        "web_sales", "web_site",
    ]
    try:
        base = f"s3a://silver/{sf}/delta/parquet"
        loaded = []
        for name in TPCDS_TABLES:
            try:
                spark.read.format("delta").load(f"{base}/{name}").createOrReplaceTempView(name)
                loaded.append(name)
            except Exception as le:
                # nao silenciar: registrar a causa real (antes um `except: pass`
                # mascarava falhas de s3a/delta e a query estourava com TABLE_NOT_FOUND)
                print(f"ADVISOR_LOAD_ERROR {name}: {str(le)[:300]}", flush=True)
        print(f"ADVISOR_LOADED {len(loaded)}/{len(TPCDS_TABLES)}: {loaded}", flush=True)
        sql = open(f"/opt/advisor/queries/{query}.sql").read().strip().rstrip(";").strip()
        # zera o acumulador logo antes do query para isolar as métricas dele dos
        # reads de metadata do Delta feitos no registro das views acima.
        for k in list(metrics):
            metrics[k] = 0
        t0 = time.time()
        spark.sql(sql).write.format("noop").mode("overwrite").save()
        dur = (time.time() - t0) * 1000
        _flush_listener_bus(spark)   # garante que os stages do query já contaram
        print(f"ADVISOR_METRICS {json.dumps(metrics)}", flush=True)
        _write_result(build_result("success", round(dur, 1), dict(metrics), None))
    except Exception as e:
        _write_result(build_result(classify_error(e), None, metrics, str(e)[:500]))
        spark.stop()
        sys.exit(1)
    spark.stop()


if __name__ == "__main__":
    # o callback server py4j do SparkListener é uma thread NÃO-daemon: após o
    # main() o processo do driver não encerraria sozinho, o pod ficaria vivo e o
    # submitter estouraria o timeout (marcando "error" apesar do sucesso já
    # gravado). Como resultado/stdout/MinIO já foram flushados no main(),
    # forçamos o término com o exit code correto.
    _rc = 0
    try:
        main()
    except SystemExit as _e:
        _rc = _e.code if isinstance(_e.code, int) else 1
    except BaseException as _e:
        print("ADVISOR_FATAL " + str(_e)[:300], flush=True)
        _rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
