import json
import os
import re
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
                "disk_spill_bytes", "records_read",
                # trabalho real (executor_cpu_time / run_time) + contagem de
                # stages/tasks: permitem derivar UTILIZACAO = cpu / (cores_tot x
                # wall), o sinal que calibra paralelismo (query leve com 96 cores
                # e util ~5% e auto-explicativa). Mesmos campos da matriz offline.
                "executor_run_time_ms", "executor_cpu_time_ms", "num_tasks", "num_stages")


def zero_metrics() -> dict:
    return {k: 0 for k in _METRIC_KEYS}


def accumulate(acc: dict, stage: dict) -> dict:
    """Soma as métricas de um stage no acumulador (chamado pelo listener por
    stage). Ignora chaves ausentes e valores None. Puro → unit-testável."""
    for k in _METRIC_KEYS:
        acc[k] = int(acc.get(k, 0)) + int(stage.get(k, 0) or 0)
    return acc


def task_stats(durations_ms: list) -> dict:
    """max/mediana das duracoes de task -> skew (max/mediana alto = straggler;
    com muitas tasks curtas o overhead de scheduling domina). Puro."""
    if not durations_ms:
        return {"max_task_ms": 0, "median_task_ms": 0}
    s = sorted(int(d or 0) for d in durations_ms)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"max_task_ms": int(s[-1]), "median_task_ms": int(med)}


# Resumo do plano fisico (mesmas features de plano da matriz offline, onde
# dominaram o XGBoost): contagem de operadores por NOME exato do operador no
# inicio da linha (apos os caracteres de arvore `+- :- *(n)`), para nao confundir
# Sort com SortMergeJoin nem Exchange com BroadcastExchange.
_PLAN_OP = re.compile(r"^[\s:+\-*()\d]*([A-Za-z]+)")
_JOINS = {"BroadcastHashJoin", "SortMergeJoin", "ShuffledHashJoin",
          "BroadcastNestedLoopJoin", "CartesianProduct"}
_BCAST_JOINS = {"BroadcastHashJoin", "BroadcastNestedLoopJoin"}
_AGGS = {"HashAggregate", "SortAggregate", "ObjectHashAggregate"}
_SCANS = {"Scan", "FileScan", "BatchScan"}


def pick_median_run(durations_ms: list, metrics_list: list) -> tuple:
    """(mediana, metricas do run mais proximo da mediana). O driver executa a
    query `TRIES` vezes NA MESMA SESSAO (protocolo da matriz offline) e reporta
    a mediana dos runs QUENTES; as metricas representativas sao as do run cuja
    duracao esta mais perto da mediana. Puro."""
    if not durations_ms:
        return None, {}
    s = sorted(durations_ms)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    i = min(range(len(durations_ms)), key=lambda k: abs(durations_ms[k] - med))
    return med, metrics_list[i]


def plan_summary(plan_text: str) -> dict:
    """Conta operadores do plano fisico (texto de executedPlan). Puro."""
    ops = []
    for line in (plan_text or "").splitlines():
        m = _PLAN_OP.match(line)
        if m:
            ops.append(m.group(1))
    return {
        "plan_joins": sum(o in _JOINS for o in ops),
        "plan_broadcast_joins": sum(o in _BCAST_JOINS for o in ops),
        "plan_sort_merge_joins": sum(o == "SortMergeJoin" for o in ops),
        "plan_exchanges": sum(o == "Exchange" for o in ops),
        "plan_aggregates": sum(o in _AGGS for o in ops),
        "plan_sorts": sum(o == "Sort" for o in ops),
        "plan_scans": sum(o in _SCANS for o in ops),
        "plan_lines": len(ops),
    }


def _register_metrics_listener(spark, acc, task_durs=None):
    """Registra um SparkListener via py4j que acumula taskMetrics por stage em
    `acc` e, se `task_durs` (lista) for dado, a duracao de cada task em ms.
    Degradação graciosa: se o callback server / addSparkListener falhar,
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
                        "executor_run_time_ms": tm.executorRunTime(),
                        "executor_cpu_time_ms": tm.executorCpuTime() // 1_000_000,  # ns -> ms
                        "num_tasks": stageCompleted.stageInfo().numTasks(),
                        "num_stages": 1,
                    })
                except Exception as ex:
                    print("ADVISOR_LISTENER_STAGE_ERROR " + str(ex)[:200], flush=True)

            def onTaskEnd(self, taskEnd):
                if task_durs is None:
                    return
                try:
                    task_durs.append(int(taskEnd.taskInfo().duration()))
                except Exception as ex:
                    print("ADVISOR_LISTENER_TASK_ERROR " + str(ex)[:200], flush=True)

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


def _wait_for_executors(spark, timeout_s=180):
    """Espera os executores REGISTRAREM antes de cronometrar. Sem isso o t0 e
    tomado enquanto pods de executor ainda sobem (JVM/imagem/scheduling) e a
    query 'paga' esse startup -- ~45s fixos por run que a matriz offline (99
    queries numa sessao quente) nunca pagou. Degradacao graciosa: segue apos o
    timeout com o que registrou."""
    n, expected, t0 = -1, 0, time.time()
    try:
        sc = spark.sparkContext
        expected = int(sc.getConf().get("spark.executor.instances", "0") or 0)
        while time.time() - t0 < timeout_s:
            n = sc._jsc.sc().getExecutorMemoryStatus().size() - 1   # -1 = driver
            if expected and n >= expected:
                break
            time.sleep(2)
    except Exception as ex:
        print("ADVISOR_EXECUTORS_ERROR " + str(ex)[:200], flush=True)
    print(f"ADVISOR_EXECUTORS {n}/{expected} registrados apos {time.time() - t0:.0f}s",
          flush=True)


def _run_once(spark, df, metrics, task_durs) -> tuple:
    """Uma execucao cronometrada: zera acumuladores, executa, drena o listener
    bus e devolve (duration_ms, metricas desta execucao)."""
    for k in list(metrics):
        metrics[k] = 0
    del task_durs[:]
    t0 = time.time()
    df.write.format("noop").mode("overwrite").save()
    dur = (time.time() - t0) * 1000
    _flush_listener_bus(spark)
    m = dict(metrics)
    m.update(task_stats(task_durs))
    return dur, m


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


TPCDS_TABLES = [
    "call_center", "catalog_page", "catalog_returns", "catalog_sales",
    "customer", "customer_address", "customer_demographics", "date_dim",
    "household_demographics", "income_band", "inventory", "item",
    "promotion", "reason", "ship_mode", "store", "store_returns",
    "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
    "web_sales", "web_site",
]
TPCH_TABLES = ["customer", "lineitem", "nation", "orders", "part",
               "partsupp", "region", "supplier"]


def resolve_dataset(dataset: str) -> dict:
    """'sf500' (TPC-DS legado) | 'tpch-sf500' -> path Silver, tabelas e dir de
    queries. Espelha advisor_loop.benchmarks (este arquivo roda solto na imagem)."""
    if dataset.startswith("tpch-"):
        sf = dataset.removeprefix("tpch-")
        return {"silver": f"s3a://silver/{sf}/tpch/delta/parquet", "tables": TPCH_TABLES,
                "queries_dir": "/opt/advisor/queries-tpch"}
    return {"silver": f"s3a://silver/{dataset}/delta/parquet", "tables": TPCDS_TABLES,
            "queries_dir": "/opt/advisor/queries"}


def main():   # coberto pela validação ao vivo, não por unit test
    from pyspark.sql import SparkSession
    sf = os.environ["SF"]
    query = os.environ["QUERY"]
    spark = SparkSession.builder.appName(f"advisor-{query}-{sf}").getOrCreate()
    # métricas reais coletadas por um SparkListener (acumulador mutado nos callbacks)
    metrics = zero_metrics()
    task_durs: list = []
    _register_metrics_listener(spark, metrics, task_durs)
    # todas as tabelas do benchmark (registrar as que existirem na Silver evita
    # a fragilidade de extrair nomes do SQL — joins por virgula/alias)
    ds = resolve_dataset(sf)
    try:
        base = ds["silver"]
        loaded = []
        for name in ds["tables"]:
            try:
                spark.read.format("delta").load(f"{base}/{name}").createOrReplaceTempView(name)
                loaded.append(name)
            except Exception as le:
                # nao silenciar: registrar a causa real (antes um `except: pass`
                # mascarava falhas de s3a/delta e a query estourava com TABLE_NOT_FOUND)
                print(f"ADVISOR_LOAD_ERROR {name}: {str(le)[:300]}", flush=True)
        print(f"ADVISOR_LOADED {len(loaded)}/{len(ds['tables'])}: {loaded}", flush=True)
        sql = open(f"{ds['queries_dir']}/{query}.sql").read().strip().rstrip(";").strip()
        df = spark.sql(sql)
        # Protocolo QUENTE (igual ao da matriz offline): espera os executores
        # registrarem, roda WARMUP execucoes nao cronometradas (a 1a e reportada
        # como cold_ms = custo fixo do harness) e depois TRIES execucoes
        # cronometradas na MESMA sessao; duration_ms = mediana dos runs quentes.
        tries = max(1, int(os.environ.get("TRIES", "1")))
        warmup = max(0, int(os.environ.get("WARMUP", "1")))
        _wait_for_executors(spark)
        cold_ms = None
        for w in range(warmup):
            d, _ = _run_once(spark, df, metrics, task_durs)
            cold_ms = d if cold_ms is None else cold_ms
            print(f"ADVISOR_WARMUP {w + 1}/{warmup} {d:.0f}ms", flush=True)
        durs, mets = [], []
        for t in range(tries):
            d, m = _run_once(spark, df, metrics, task_durs)
            durs.append(round(d, 1))
            mets.append(m)
            print(f"ADVISOR_TRY {t + 1}/{tries} {d:.0f}ms", flush=True)
        dur, rep = pick_median_run(durs, mets)
        # plano fisico FINAL (com AQE, executedPlan apos a execucao e o plano
        # adaptado): resumo em contagens no metrics; texto bruto so no stdout.
        try:
            plan_txt = df._jdf.queryExecution().executedPlan().toString()
            rep.update(plan_summary(plan_txt))
            print("ADVISOR_PLAN " + plan_txt[:4000].replace("\n", "\\n"), flush=True)
        except Exception as pe:
            print("ADVISOR_PLAN_ERROR " + str(pe)[:200], flush=True)
        print(f"ADVISOR_METRICS {json.dumps(rep)}", flush=True)
        result = build_result("success", round(dur, 1), rep, None)
        result["durations_ms"] = durs
        result["cold_ms"] = round(cold_ms, 1) if cold_ms is not None else None
        _write_result(result)
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
