# advisor-loop Closed Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fechar o loop do advisor-loop com execução Spark real: o advisor propõe uma config, o Spark roda a query no cluster com ela, a duração + métricas de I/O reais voltam como feedback, e o advisor re-propõe iterando até convergir.

**Architecture:** Um `SparkK8sBackend` (ExecutionBackend) usa um `SparkSubmitter` que renderiza um launcher pod (imagem dedicada `spark-advisor` com driver+queries), aplica via SSH→kubectl, espera, e lê o resultado (status+métricas) do MinIO com fallback nos logs. Um `OptimizationLoop` orquestra propose→validate→execute→distill→re-propose com parada por N iterações ou sem-melhora, alimentando o histórico de volta ao `Advisor.advise(history=...)`.

**Tech Stack:** Python ≥3.11, pydantic v2, PySpark 3.5 + Delta (na imagem), DuckDB (probes existentes), subprocess/SSH para kubectl, GHCR (imagem), pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-advisor-loop-closed-loop-design.md` (traga para o repo advisor-loop junto com este plano)

## Global Constraints

- Trabalhar no repo **`advisor-loop`** (branch nova `feat/closed-loop` a partir de `main`). Manter suíte verde; TDD (teste falha → implementa → passa → commit). Não fazer push nem merge (controller decide).
- Testes NUNCA tocam o cluster nem fazem SSH/subprocess reais: SparkSubmitter e SparkK8sBackend são testados com subprocess/SSH **mockado**; OptimizationLoop com **backend fake**.
- Interface `ExecutionBackend.execute(episode, rec) -> Outcome` é a mesma de ReplayBackend/ProposalBackend — o advisor/runner não muda de forma.
- Confs Spark fixas (padrão exp05, verbatim): master `k8s://https://kubernetes.default.svc:443`; `spark.kubernetes.namespace=italo-unb`; `spark.kubernetes.authenticate.driver.serviceAccountName=ns-admin`; `spark.hadoop.fs.s3a.endpoint=http://minio:9000`; `spark.hadoop.fs.s3a.access.key=minioadmin`; `spark.hadoop.fs.s3a.secret.key=minioadmin`; `spark.hadoop.fs.s3a.path.style.access=true`; `spark.hadoop.fs.s3a.connection.ssl.enabled=false`; `spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension`; `spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog`.
- ConfigVector→Spark: executor_cores→spark.executor.cores; executor_memory_gb→spark.executor.memory={N}g; executor_instances→spark.executor.instances; driver_memory_gb→spark.driver.memory={N}g; shuffle_partitions→spark.sql.shuffle.partitions; advisory_partition_mb→spark.sql.adaptive.advisoryPartitionSizeInBytes={N}m; aqe_enabled→spark.sql.adaptive.enabled; broadcast_threshold_mb→spark.sql.autoBroadcastJoinThreshold={N}m.
- Imagem dedicada: `ghcr.io/italovinicius18/spark-advisor:3.5.0-delta`, pública, com driver + queries bundled.
- 1 job por vez (anti-contaminação); timeout duro por job; limpeza de pods garantida (finally).

---

### Task 1: `QueryResult.metrics` + status `proposed` já existe — estender o modelo

**Files:**
- Modify: `advisor_loop/episode.py`
- Test: `tests/test_episode.py`

**Interfaces:**
- Produces: `QueryResult` ganha `metrics: dict = {}` (input_bytes, output_bytes, shuffle_read_bytes, shuffle_write_bytes, gc_time_ms, read_throughput_mbps, write_throughput_mbps). `Outcome.status` ganha o literal `"oom"` (além dos existentes). Nada mais muda.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_episode.py — adicionar
from advisor_loop.episode import QueryResult, Outcome

def test_queryresult_carries_metrics():
    qr = QueryResult(status="success", duration_ms=1200.0,
                     metrics={"input_bytes": 100, "read_throughput_mbps": 5.0})
    assert qr.metrics["input_bytes"] == 100
    assert QueryResult(status="failed").metrics == {}   # default vazio

def test_outcome_status_oom_allowed():
    Outcome(status="oom")   # não levanta
```

- [ ] **Step 2: Rodar e ver falhar** — `PYTHONPATH=. .venv/bin/pytest tests/test_episode.py -k "metrics or oom" -v` → FAIL (metrics não existe / oom inválido)

- [ ] **Step 3: Implementar** — em `episode.py`:
- Adicionar `metrics: dict = Field(default_factory=dict)` em `QueryResult`.
- Adicionar `"oom"` ao Literal de `QueryResult.status` e ao de `Outcome.status`.

- [ ] **Step 4: Rodar e ver passar** → PASS; rodar a suíte toda (`PYTHONPATH=. .venv/bin/pytest -q`).
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: QueryResult.metrics + oom status for real-execution outcomes"`

---

### Task 2: Driver PySpark minimalista (`spark_advisor/driver.py`)

**Files:**
- Create: `spark_advisor/driver.py`, `spark_advisor/__init__.py`
- Test: `tests/test_driver_helpers.py`

**Interfaces:**
- Produces: um script executável `driver.py` que, dado env `SF, QUERY, RUN_ID, RESULT_S3` + confs Spark (via spark-submit), roda a query e escreve o JSON de resultado. Funções puras testáveis SEM Spark: `build_result(status, duration_ms, metrics, error) -> dict`, `classify_error(exc_or_text) -> str` (retorna "oom"|"timeout"|"failed"), `throughput_mbps(bytes_, duration_ms) -> float`.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_driver_helpers.py
from spark_advisor.driver import build_result, classify_error, throughput_mbps

def test_throughput():
    assert throughput_mbps(10_000_000, 1000.0) == 10.0   # 10MB em 1s = 10 MB/s
    assert throughput_mbps(0, 0) == 0.0

def test_classify_error():
    assert classify_error("java.lang.OutOfMemoryError: Java heap space") == "oom"
    assert classify_error("ExecutorLostFailure ... exit code 137") == "oom"
    assert classify_error("some other failure") == "failed"

def test_build_result():
    r = build_result("success", 1200.0, {"input_bytes": 5}, None)
    assert r["status"] == "success" and r["duration_ms"] == 1200.0 and r["metrics"]["input_bytes"] == 5
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar** — `spark_advisor/driver.py`:

```python
import json, os, sys, time

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
    sf = os.environ["SF"]; query = os.environ["QUERY"]
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
        spark.stop(); sys.exit(1)
    spark.stop()

if __name__ == "__main__":
    main()
```

Nota: a coleta fina de métricas via `QueryExecutionListener` é best-effort; o esqueleto acima grava as chaves (zeradas se indisponíveis). O enriquecimento das métricas de I/O é validado/afinado na fase live (Task 9). As funções puras (`throughput_mbps`, `classify_error`, `build_result`) são o que os unit tests cobrem.

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: minimal PySpark driver (pure helpers tested; main covered live)"`

---

### Task 3: Tradução ConfigVector→Spark confs (`backends/spark_conf.py`)

**Files:**
- Create: `advisor_loop/backends/spark_conf.py`
- Test: `tests/test_spark_conf.py`

**Interfaces:**
- Consumes: `ConfigVector` (episode.py).
- Produces: `spark_confs(config: ConfigVector) -> list[str]` retornando os pares `--conf spark.X=Y` (só a parte variável, sem as fixas), na ordem determinística das chaves.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_spark_conf.py
from advisor_loop.backends.spark_conf import spark_confs
from advisor_loop.episode import ConfigVector

def _cv(**o):
    base = dict(executor_cores=8, executor_memory_gb=28, executor_instances=6,
                driver_memory_gb=8, shuffle_partitions=200, advisory_partition_mb=64,
                aqe_enabled=True, broadcast_threshold_mb=10)
    return ConfigVector(**{**base, **o})

def test_translation():
    c = spark_confs(_cv())
    s = " ".join(c)
    assert "spark.executor.cores=8" in s
    assert "spark.executor.memory=28g" in s
    assert "spark.executor.instances=6" in s
    assert "spark.driver.memory=8g" in s
    assert "spark.sql.shuffle.partitions=200" in s
    assert "spark.sql.adaptive.advisoryPartitionSizeInBytes=64m" in s
    assert "spark.sql.adaptive.enabled=true" in s
    assert "spark.sql.autoBroadcastJoinThreshold=10m" in s

def test_broadcast_off():
    assert "spark.sql.autoBroadcastJoinThreshold=-1" in " ".join(spark_confs(_cv(broadcast_threshold_mb=-1)))
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/backends/spark_conf.py
from advisor_loop.episode import ConfigVector

def spark_confs(c: ConfigVector) -> list[str]:
    bcast = "-1" if c.broadcast_threshold_mb == -1 else f"{c.broadcast_threshold_mb}m"
    pairs = {
        "spark.executor.cores": c.executor_cores,
        "spark.executor.memory": f"{c.executor_memory_gb}g",
        "spark.executor.instances": c.executor_instances,
        "spark.driver.memory": f"{c.driver_memory_gb}g",
        "spark.sql.shuffle.partitions": c.shuffle_partitions,
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": f"{c.advisory_partition_mb}m",
        "spark.sql.adaptive.enabled": str(c.aqe_enabled).lower(),
        "spark.sql.autoBroadcastJoinThreshold": bcast,
    }
    out = []
    for k in pairs:  # dict preserva ordem de inserção → determinístico
        out += ["--conf", f"{k}={pairs[k]}"]
    return out
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: ConfigVector -> spark-submit conf translation"`

---

### Task 4: Launcher manifest render (`backends/launcher.py`)

**Files:**
- Create: `advisor_loop/backends/launcher.py`
- Test: `tests/test_launcher.py`

**Interfaces:**
- Consumes: `spark_confs` (Task 3), `ConfigVector`.
- Produces: `render_launcher(config, sf, query, run_id, image, result_s3) -> str` — YAML de um Pod k8s (namespace italo-unb, SA ns-admin, imagem dada) cujo container roda `spark-submit` com as confs fixas + variáveis + o driver + env (SF, QUERY, RUN_ID, RESULT_S3, MINIO_*). Nome do pod = `advisor-{run_id}`. `restartPolicy: Never`.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_launcher.py
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
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar** — `render_launcher` monta um dict e serializa com `yaml.safe_dump`. O container `command: ["/opt/spark/bin/spark-submit"]`, `args` = confs fixas (as do Global Constraints, incl. master, image `spark.kubernetes.container.image={image}`, delta, s3a) + `spark_confs(config)` + `local:///opt/advisor/driver.py`. `env`: SF, QUERY, RUN_ID, RESULT_S3, MINIO_ENDPOINT=http://minio:9000, MINIO_ACCESS_KEY=minioadmin, MINIO_SECRET_KEY=minioadmin. Namespace italo-unb, SA ns-admin, restartPolicy Never. (Escrever a lista de confs fixas verbatim do Global Constraints.)

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: k8s launcher-pod manifest render for spark-submit"`

---

### Task 5: `SparkSubmitter` (SSH→kubectl, poll, resultado) — subprocess mockado

**Files:**
- Create: `advisor_loop/backends/spark_submitter.py`
- Test: `tests/test_spark_submitter.py`

**Interfaces:**
- Consumes: `render_launcher` (Task 4).
- Produces: `SparkSubmitter(ssh_cmd: list[str], image, result_bucket="advisor-results", poll_interval_s=10, max_wall_clock_s=1200, runner=subprocess.run)` com `submit(config, sf, query, run_id) -> dict` = `{status, duration_ms, metrics, error}`. Fluxo: `kubectl apply -f -` (manifesto no stdin) → poll `kubectl get pod advisor-{run_id} -o jsonpath phase` até Succeeded/Failed/timeout → ler resultado (via `kubectl logs` procurando `ADVISOR_RESULT `, com fallback via `mc cat` do MinIO — no primeiro corte, logs bastam) → `kubectl delete pod` (finally). `runner` injetável = wrapper sobre `ssh_cmd + [remote_cmd]`; testes injetam um fake. Ausência de `ADVISOR_RESULT` mas pod Succeeded → status `failed` (driver crash mascarado).

- [ ] **Step 1: Teste falhando**

```python
# tests/test_spark_submitter.py
import json
from advisor_loop.backends.spark_submitter import SparkSubmitter
from tests.test_spark_conf import _cv

class FakeRunner:
    """Simula ssh→kubectl: guarda comandos, devolve saídas roteirizadas por etapa."""
    def __init__(self, phases, logs):
        self.phases = list(phases); self.logs = logs; self.calls = []
    def __call__(self, cmd, **kw):
        import types
        self.calls.append(cmd)
        joined = " ".join(cmd)
        out = ""
        if "apply" in joined: out = "pod/advisor-r1 created"
        elif "jsonpath" in joined or "get pod" in joined: out = self.phases.pop(0)
        elif "logs" in joined: out = self.logs
        elif "delete" in joined: out = "deleted"
        return types.SimpleNamespace(stdout=out, returncode=0)

def test_submit_success_reads_result():
    logs = "some spark noise\nADVISOR_RESULT " + json.dumps(
        {"status":"success","duration_ms":4300.0,"metrics":{"input_bytes":9},"error":None}) + "\nmore"
    r = FakeRunner(phases=["Pending","Running","Succeeded"], logs=logs)
    s = SparkSubmitter(ssh_cmd=["ssh","host"], image="img", runner=r, poll_interval_s=0)
    res = s.submit(_cv(), "sf1", "q03", "r1")
    assert res["status"] == "success" and res["duration_ms"] == 4300.0
    assert any("delete" in " ".join(c) for c in r.calls)   # limpeza chamada

def test_succeeded_without_result_is_failed():
    r = FakeRunner(phases=["Succeeded"], logs="no marker here")
    s = SparkSubmitter(ssh_cmd=["ssh","host"], image="img", runner=r, poll_interval_s=0)
    assert s.submit(_cv(), "sf1", "q03", "r2")["status"] == "failed"

def test_pod_failed_is_failed():
    r = FakeRunner(phases=["Running","Failed"], logs="java.lang.OutOfMemoryError")
    s = SparkSubmitter(ssh_cmd=["ssh","host"], image="img", runner=r, poll_interval_s=0)
    res = s.submit(_cv(), "sf1", "q03", "r3")
    assert res["status"] in ("failed","oom")
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar** — `spark_submitter.py`: `_run(remote_cmd)` = `self.runner(self.ssh_cmd + [remote_cmd])`; `apply` passa o manifesto via `kubectl apply -f -` (o manifesto entra no comando remoto por heredoc/stdin — no fake é só a presença de "apply"); poll com `time.sleep(poll_interval_s)` até phase terminal ou `max_wall_clock_s` (usar um relógio injetável ou contar iterações; nos testes poll_interval_s=0); ao terminar, `kubectl logs` → procurar `ADVISOR_RESULT ` → parse JSON; se Succeeded sem marcador → `{"status":"failed", ...}`; se Failed → parse do log via `classify_error` (importar de spark_advisor.driver) para oom/failed; `finally: kubectl delete pod`. Retorna o dict de resultado. **Não** dormir de verdade nos testes (poll_interval_s=0 e phases termina rápido).

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: SparkSubmitter (ssh->kubectl apply/poll/result/cleanup), subprocess-mocked"`

---

### Task 6: `SparkK8sBackend` (ExecutionBackend)

**Files:**
- Create: `advisor_loop/backends/spark_k8s.py`
- Test: `tests/test_spark_k8s_backend.py`

**Interfaces:**
- Consumes: `SparkSubmitter` (Task 5), `Episode/Recommendation/Outcome/QueryResult` (episode.py).
- Produces: `SparkK8sBackend(submitter, run_id_fn=...)` com `execute(episode, rec) -> Outcome`. Para cada query do episódio (no primeiro corte, 1 query): gera run_id, chama `submitter.submit(rec.config, episode.dataset, query, run_id)`, mapeia o dict → `QueryResult(status, duration_ms, metrics, error)`. `Outcome.status` = "completed" se todas success; senão o status da falha (failed/oom/timeout). `Outcome.wall_clock_s` = soma das durações.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_spark_k8s_backend.py
from advisor_loop.backends.spark_k8s import SparkK8sBackend
from advisor_loop.episode import Episode, Recommendation, ConfigVector

class FakeSubmitter:
    def __init__(self, result): self.result = result; self.seen = []
    def submit(self, config, sf, query, run_id):
        self.seen.append((sf, query, run_id)); return self.result

def _rec():
    return Recommendation(rationale="", config=ConfigVector(
        executor_cores=8, executor_memory_gb=28, executor_instances=6, driver_memory_gb=8,
        shuffle_partitions=200, advisory_partition_mb=64, aqe_enabled=True, broadcast_threshold_mb=10))

def test_execute_success_maps_outcome():
    sub = FakeSubmitter({"status":"success","duration_ms":4300.0,"metrics":{"input_bytes":9},"error":None})
    be = SparkK8sBackend(sub, run_id_fn=lambda: "rid")
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    out = be.execute(ep, _rec())
    assert out.status == "completed"
    assert out.per_query["q03"].duration_ms == 4300.0
    assert out.per_query["q03"].metrics["input_bytes"] == 9
    assert sub.seen == [("sf1","q03","rid")]

def test_execute_oom_propagates():
    sub = FakeSubmitter({"status":"oom","duration_ms":None,"metrics":{},"error":"OOM"})
    be = SparkK8sBackend(sub, run_id_fn=lambda: "rid")
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    out = be.execute(ep, _rec())
    assert out.status == "oom" and out.per_query["q03"].status == "oom"
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar** — `spark_k8s.py`: iterar `episode.queries`, `submit`, montar `QueryResult`; `Outcome.status` = "completed" se todos success, senão o primeiro status não-success; `wall_clock_s` = soma de duration_ms/1000 (None→0). `run_id_fn` default gera de episode+query+uuid4 (mas nos testes é injetado).

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: SparkK8sBackend mapping submitter results to Outcome"`

---

### Task 7: `Advisor.advise(history=...)` — feedback das iterações

**Files:**
- Modify: `advisor_loop/advisor/advisor.py`
- Test: `tests/test_advisor.py`

**Interfaces:**
- Consumes: `ConfigVector`, `Outcome` (episode.py).
- Produces: `advise(ep, context, probe, memories, history=[])` — novo param `history: list[tuple[ConfigVector, Outcome]] = []`. Quando não-vazio, injeta no contexto (junto da lista de tabelas/orçamento) um bloco `render_history(history)` com linhas `"iter N: {c}c/{m}g/{w}w → {dur}ms"` ou `"→ {status}"` (falha). Adicionar helper module-level `render_history(history) -> str`.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_advisor.py — adicionar
from advisor_loop.advisor.advisor import Advisor, render_history
from advisor_loop.episode import ConfigVector, Outcome, QueryResult, Episode
from advisor_loop.advisor.provider import LLMResponse
import json

CFG = dict(executor_cores=8, executor_memory_gb=28, executor_instances=6, driver_memory_gb=8,
           shuffle_partitions=200, advisory_partition_mb=64, aqe_enabled=True, broadcast_threshold_mb=10)

def test_render_history_lines():
    cv = ConfigVector(**CFG)
    hist = [(cv, Outcome(status="completed", per_query={"q":QueryResult(status="success",duration_ms=4300.0)})),
            (ConfigVector(**{**CFG,"executor_instances":12}), Outcome(status="oom"))]
    s = render_history(hist)
    assert "4300" in s and "oom" in s and "12w" in s

class RecClient:
    def __init__(self): self.msgs=None
    def chat(self, model, messages, tools=None, temperature=0.2):
        self.msgs=messages
        args=json.dumps({**CFG,"rationale":"ok"})
        return LLMResponse(message={}, tool_calls=[{"id":"1","function":{"name":"submit_recommendation","arguments":args}}])

def test_history_injected_into_context():
    c=RecClient(); adv=Advisor(c, __import__("advisor_loop.journal",fromlist=["Journal"]).Journal("/tmp/h.jsonl"))
    ep=Episode(workload="w",dataset="sf1",queries=["q03"],provider="m")
    cv=ConfigVector(**CFG)
    adv.advise(ep,"ctx",None,[],history=[(cv,Outcome(status="completed",per_query={"q03":QueryResult(status="success",duration_ms=4300.0)}))])
    user_msg=" ".join(m.get("content","") or "" for m in c.msgs if m["role"]=="user")
    assert "4300" in user_msg
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar** — em advisor.py: adicionar `render_history(history)` (module-level) e o param `history=[]` em `advise`; se `history`, concatenar ao `context` a linha `"\n\n## Iterações anteriores nesta query (min_time)\n{render_history(history)}\nProponha uma config que REDUZA o tempo; se alguma falhou por OOM, ajuste memória/paralelismo."`. Não quebrar chamadas existentes (default []).

- [ ] **Step 4: Rodar e ver passar** → PASS; suíte toda.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: advise(history) feeds real prior results back to the advisor"`

---

### Task 8: `OptimizationLoop` (o laço)

**Files:**
- Create: `advisor_loop/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `Advisor.advise(history=)` (Task 7), Guardian, backend `.execute` (Task 6), MemoryStore, Journal, `render_history`.
- Produces: `optimize(episode, deps, advisor, max_iters=4, patience=2) -> dict` = `{best_config, best_duration_ms, iterations: [...], history}`. Laço conforme spec §2.5: por iteração — advise(history) → guardian.validate → (se ok) backend.execute → journal query_result/episode_iter + memory.distill → atualiza best/no_improve → break em patience/max_iters. `deps` reusa a dataclass `Deps` do runner (client/journal/pack/guardian/backend/memory/probe_factory). Retorna best (ou best=None se nenhuma success).

- [ ] **Step 1: Teste falhando**

```python
# tests/test_loop.py
from advisor_loop.loop import optimize
from advisor_loop.episode import ConfigVector, Outcome, QueryResult, Episode, Recommendation
from advisor_loop.guardian import Guardian
from advisor_loop.journal import Journal
from advisor_loop.memory.store import MemoryStore

CLUSTER={"max_cores_per_container":24,"max_memory_gb_per_container":96,"max_pods":20,"total_cores":100,"total_memory_gb":400}
CFG=dict(executor_cores=8,executor_memory_gb=28,executor_instances=6,driver_memory_gb=8,shuffle_partitions=200,advisory_partition_mb=64,aqe_enabled=True,broadcast_threshold_mb=10)

class ScriptAdvisor:
    """devolve configs/durações roteirizadas por iteração."""
    def __init__(self, durations): self.durations=list(durations); self.i=0
    def advise(self, ep, ctx, probe, mems, history=None):
        self.i+=1
        return Recommendation(rationale="", config=ConfigVector(**{**CFG,"executor_instances":5+self.i}))

class ScriptBackend:
    def __init__(self, durations): self.durations=list(durations)
    def execute(self, ep, rec):
        d=self.durations.pop(0)
        if d is None: return Outcome(status="oom")
        return Outcome(status="completed", per_query={ep.queries[0]:QueryResult(status="success",duration_ms=d)}, wall_clock_s=d/1000)

class Pack:
    hash="h"; 
    def render(self,ep): return "ctx"

def _deps(tmp_path, backend):
    from advisor_loop.runner import Deps
    return Deps(client=None, journal=Journal(tmp_path/"j.jsonl"), pack=Pack(),
               guardian=Guardian(CLUSTER), backend=backend, memory=MemoryStore(tmp_path/"m.jsonl"),
               probe_factory=lambda ep: None)

def test_stops_on_no_improvement(tmp_path):
    ep=Episode(workload="w",dataset="sf1",queries=["q03"],provider="m")
    be=ScriptBackend([5000, 4000, 4200, 4300])   # melhora, depois piora 2x → para
    r=optimize(ep, _deps(tmp_path,be), ScriptAdvisor([]), max_iters=4, patience=2)
    assert r["best_duration_ms"]==4000
    assert len(r["iterations"])==3   # iter3 e iter4? para após 2 sem melhora (iter3=4200,iter4? não roda)

def test_stops_on_max_iters(tmp_path):
    ep=Episode(workload="w",dataset="sf1",queries=["q03"],provider="m")
    be=ScriptBackend([5000,4900,4800,4700,4600])
    r=optimize(ep, _deps(tmp_path,be), ScriptAdvisor([]), max_iters=3, patience=2)
    assert len(r["iterations"])==3 and r["best_duration_ms"]==4800

def test_oom_does_not_crash(tmp_path):
    ep=Episode(workload="w",dataset="sf1",queries=["q03"],provider="m")
    be=ScriptBackend([None, 4000, 4100, 4200])
    r=optimize(ep, _deps(tmp_path,be), ScriptAdvisor([]), max_iters=4, patience=2)
    assert r["best_duration_ms"]==4000
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar** — `loop.py::optimize`: manter `history=[]`, `best=(None,None)`, `no_improve=0`, `iterations=[]`. Por iter: `rec=advisor.advise(ep, deps.pack.render(ep), probe, mems, history=history)`; `ok,reason=deps.guardian.validate(rec.config)`; se não ok → registra iteração inválida, `history.append((rec.config, Outcome(status="invalid_recommendation")))`, `no_improve+=1`, checa parada, continue; senão `out=deps.backend.execute(ep,rec)`; journal `{"kind":"opt_iter",...}`; `deps.memory.distill(ep,rec,out)`; extrair `dur` da primeira query success (senão None); `history.append((rec.config,out))`; se success e (best None ou dur<best_dur): best=(rec.config,dur); no_improve=0; senão no_improve+=1; `iterations.append({...})`; se no_improve>=patience ou iter==max_iters: break. Return `{"best_config": best[0].model_dump() if best[0] else None, "best_duration_ms": best[1], "iterations": iterations, "history_len": len(history)}`. (Ajustar a contagem de `iterations` para casar com os asserts: registrar UMA entrada por iteração executada.)

- [ ] **Step 4: Rodar e ver passar** → PASS (ajustar contagens se os asserts do teste precisarem — os testes são a fonte da verdade do comportamento de parada).
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: OptimizationLoop (propose->execute->feedback->re-propose, stop on N or no-improvement)"`

---

### Task 9: Wiring + campanha + Dockerfile da imagem + smoke ao vivo + PR

**Files:**
- Modify: `advisor_loop/wiring.py`, `advisor_loop/runner.py`
- Create: `docker/spark-advisor/Dockerfile`, `campaigns/closed-loop-sf1.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: tudo acima.
- Produces: (a) `wiring.build_deps` reconhece `backend: spark_k8s` (constrói `SparkK8sBackend(SparkSubmitter(ssh_cmd=<do env/datasets>, image=<do datasets>))`); (b) runner reconhece campanhas com `mode: optimize` → chama `optimize()` por query em vez do ciclo single-shot; (c) Dockerfile da imagem `spark-advisor` (Spark 3.5 + Delta jars + boto3 + COPY driver.py + queries/); (d) campanha `closed-loop-sf1.yaml`.

- [ ] **Step 1: Wiring test (fake, sem cluster)**

```python
# tests/test_wiring.py — adicionar
def test_wiring_selects_spark_k8s(tmp_path, monkeypatch):
    import yaml
    from pathlib import Path
    (tmp_path/"datasets.yaml").write_text(yaml.safe_dump({
        "spark": {"ssh_cmd": ["ssh","host"], "image": "ghcr.io/x/spark-advisor:t"}, "sf1": {}}))
    (tmp_path/"packs").mkdir(); (tmp_path/"packs"/"cluster.yaml").write_text(
        "cluster: {max_cores_per_container: 24, max_memory_gb_per_container: 96, max_pods: 20, total_cores: 100, total_memory_gb: 400}")
    (tmp_path/"campaigns").mkdir()
    camp=tmp_path/"campaigns"/"c.yaml"; camp.write_text(yaml.safe_dump({"name":"c","level":"L1","provider":"m","backend":"spark_k8s","datasets":["sf1"],"queries":["q03"],"tries":1,"memory":"off","mode":"optimize"}))
    (tmp_path/"data").mkdir(); (tmp_path/"data"/"reference.parquet").touch()
    from advisor_loop.wiring import build_deps
    deps=build_deps(camp, root=tmp_path)
    from advisor_loop.backends.spark_k8s import SparkK8sBackend
    assert isinstance(deps.backend, SparkK8sBackend)
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar wiring + runner mode** — wiring: se `camp["backend"]=="spark_k8s"`, ler bloco `spark:` do datasets.yaml (`ssh_cmd`, `image`), construir `SparkK8sBackend(SparkSubmitter(ssh_cmd, image))`. runner: se `camp.get("mode")=="optimize"`, para cada (dataset,query) chamar `optimize(ep, deps, Advisor(deps.client, deps.journal), max_iters=camp.get("max_iters",4), patience=camp.get("patience",2))` e journal o resultado; senão o fluxo atual. (Não quebrar campanhas existentes: mode default = single-shot.)

- [ ] **Step 4: Rodar e ver passar** → PASS; suíte toda verde.

- [ ] **Step 5: Dockerfile + campanha (sem build no unit test)**

```dockerfile
# docker/spark-advisor/Dockerfile
FROM apache/spark:3.5.0
USER root
RUN pip install --no-cache-dir boto3 delta-spark==3.2.0
# jars Delta p/ o Spark (delta-spark instala os py; os jars vêm via --packages ou COPY)
COPY spark_advisor/driver.py /opt/advisor/driver.py
COPY queries/ /opt/advisor/queries/
ENV PYTHONPATH=/opt/advisor
USER spark
```

```yaml
# campaigns/closed-loop-sf1.yaml
name: closed-loop-sf1
mode: optimize
level: L1
provider: deepseek/deepseek-v4-flash-0731
backend: spark_k8s
datasets: [sf1]
queries: [q03]
max_iters: 4
patience: 2
memory: "on"
```

Nota: os jars Delta precisam estar disponíveis ao spark-submit (via `--packages io.delta:delta-spark_2.12:3.2.0` nas confs fixas do launcher, OU pré-baixados na imagem). O launcher (Task 4) já inclui as confs de extensão Delta; adicionar `--packages io.delta:delta-spark_2.12:3.2.0` às confs fixas do launcher para o download em runtime (ou documentar o COPY dos jars). Validar na fase live.

- [ ] **Step 6: Commit do código** — `git add -A && git commit -m "feat: wiring spark_k8s + optimize mode + spark-advisor Dockerfile + closed-loop campaign"`

- [ ] **Step 7: Smoke ao vivo (controller)** — build+push da imagem `ghcr.io/italovinicius18/spark-advisor:3.5.0-delta` (docker build docker/spark-advisor + push, imagem pública); rodar `advisor-loop run campaigns/closed-loop-sf1.yaml` com o tunnel/SSH configurado; verificar: pod launcher aplica, driver roda q03 em SF1, resultado (duração+métricas) volta, o loop re-propõe vendo o histórico, best config retornada; medir custo (cluster + API). Registrar evidência.

- [ ] **Step 8: README + PR** — documentar o modo optimize + a imagem; `gh pr create --draft` da branch `feat/closed-loop`.

---

## Self-Review (executada)

1. **Cobertura da spec**: §2.1 componentes → Tasks 5 (Submitter), 6 (Backend), 8 (Loop), 7 (advise history), 1 (QueryResult.metrics). §2.2 tradução → Task 3. §2.3 driver+métricas+MinIO → Task 2 (+ live tune Task 9). §2.4 submit SSH→kubectl+poll+result+cleanup → Tasks 4-5. §2.5 laço/parada → Task 8. §3 feedback → Task 7. §4 erro/segurança (guardian, timeout, 1-por-vez, cleanup, crash-mascarado) → Tasks 5,6,8. §5 testes → cada task. §7 imagem dedicada GHCR + MinIO+logs → Tasks 2,9. §6 marcos F1-F3 → Tasks 2-5 (F1), 6-9 (F2), 9 (F3).
2. **Placeholders**: nenhum TBD; todo step tem código/comando. As partes validadas só ao vivo (driver.main, jars Delta, métricas finas) estão explicitamente marcadas como live-tune (Task 9), não como placeholder — as funções puras têm testes.
3. **Consistência de tipos**: `spark_confs`(T3)→`render_launcher`(T4)→`SparkSubmitter.submit`(T5)→`SparkK8sBackend.execute`(T6); dict de resultado `{status,duration_ms,metrics,error}` consistente T2/T5/T6; `render_history`/`advise(history=)`(T7) usados por `optimize`(T8); `Deps` reusada de runner.py em T8. `QueryResult.metrics` (T1) preenchido em T6.

---

## Validação ao vivo (2026-08-23) — CONCLUÍDA

Fluxo fechado **validado ponta a ponta** no cluster RNP (SF1/q03), com uso leve
para não contaminar o exp05 em curso (config pequena, nó ocioso `ids-pb`).

### Migração para Spark 4.2 (decisão do usuário)
A imagem migrou de Spark 3.5/Delta 3.2 para **Spark 4.2.0 + Delta 4.4.0** (Scala
2.13, Java 17) — par oficial (Delta 4.4.0 compila contra spark-sql 4.2.0). A
lógica do advisor/loop é agnóstica de versão; só imagem + confs + driver mudaram.
Imagem: `ghcr.io/italovinicius18/spark-advisor:4.2.0-delta-r3` (pública, GHCR).

### Achados resolvidos na validação (não capturáveis por unit test)
1. **Ivy em runtime falha no pod** (`spark.jars.packages` → cache não-gravável /
   egress). Fix: **assar todos os JARs na imagem** (Delta 4.4 + delta-storage +
   hadoop-aws 3.5.0 + AWS SDK v2 bundle 2.35.4 + **analyticsaccelerator-s3 1.3.1**,
   dependência obrigatória do S3A no Hadoop 3.5).
2. **deploy-mode cluster**: driver roda em pod separado → env não chega e o
   `ADVISOR_RESULT` sai no log do DRIVER. Fix: `driverEnv.*` + `driver.pod.name`
   previsível + submitter lê/limpa o pod driver.
3. **AWS SDK v2 (Hadoop 3.5)** exige `fs.s3a.endpoint.region` mesmo p/ MinIO.
4. **ANSI on por padrão no Spark 4** quebraria a Silver toda-VARCHAR (escrita em
   3.5). Fix: `spark.sql.ansi.enabled=false` (mantém semântica de origem).
5. **DNS ao ghcr.io intermitente** em alguns nós + imagem só cacheada em alguns.
   Fix: `pullPolicy=IfNotPresent` + **`node_selector`** opcional (via datasets.yaml)
   fixando driver+executors num nó com egress/imagem — também isola do exp05.
6. **`subprocess.run` real** precisa `capture_output=True, text=True` (o fake
   runner dos testes mascarava; o `.stdout=None` só quebrou ao vivo). Fix + teste
   de regressão.
7. **Driver silenciava falhas de load** (`except: pass`) → mascarava a causa e a
   query estourava `TABLE_NOT_FOUND`. Fix: logar `ADVISOR_LOAD_ERROR`/`ADVISOR_LOADED`.

### Resultado (campaign `closed-loop-sf1`, 2 iterações)
- iter 1: q03 SF1 = **19903.8 ms** (success, 24/24 tabelas)
- iter 2: **19507.1 ms** (best) — advisor re-propôs vendo o histórico
- `opt_result.best_duration_ms = 19507.1`; loop escolheu a melhor config
- Custo LLM real ~US$0.0004 (deepseek-v4-flash); cluster limpo após o run
- Ambos os caminhos de resultado validados: marcador `ADVISOR_RESULT ` no stdout
  do driver **e** objeto JSON no MinIO (`s3://advisor-results/<run_id>.json`)

### Métricas reais no histórico do LLM (2026-08-23, follow-up)
Antes o LLM decidia a próxima iteração **só pelo tempo** (o `render_history` só
expunha `duração`/`status`), e o dict de métricas do driver era zerado. Corrigido:
- **SparkListener via py4j** no driver (`_register_metrics_listener`) acumula
  `taskMetrics` por stage: input/output bytes, shuffle read/write, spill
  (mem+disk), GC, records. Zerado logo antes do query p/ isolar dos reads de
  metadata do Delta; `listenerBus().waitUntilEmpty` antes de ler. Degradação
  graciosa (nunca derruba o job).
- Gotcha resolvido: `start_callback_server()` sobe numa porta aleatória mas o
  Java mantinha o CallbackClient na porta default → `Connection refused` e
  métricas zeradas. Fix: `resetCallbackClient(addr, porta_real)`.
- `render_history` agora injeta as métricas na linha de cada iteração, ex.:
  `iter 1: 2c/4g/1w → 34137.6ms (in=470.4KB, shuffle_r=57.9KB, shuffle_w=57.9KB, gc=383ms)`.
- Validado ao vivo (SF1/q03): `input_bytes=481690, records_read=2971504,
  shuffle=59331, gc=383ms`, 0 erros de listener.

### Pendências conhecidas (documentadas, não bloqueantes)
- `node_selector` de nó único limita paralelismo à capacidade do nó; para escalas
  maiores, rotular N nós com egress e fixar pelo label (não por hostname único).
- `cost_usd` no summary é 0 (custo real fica nos eventos `llm_call` do journal).

### Piloto SF1 completo com métricas no histórico (2026-08-23)
4 iterações, todas success (o fix de término do driver eliminou o falso "error"
por timeout do submitter). Métricas reais agora persistidas no journal por iteração
e realimentadas ao LLM via `render_history`:
- iter 1: 8×8c/28g → 20051ms (shuffle_w=114KB, gc=3657ms)
- iter 2: 4×8c/28g → 19906ms (shuffle_w=124KB, gc=3288ms)
- iter 3: 2×4c/14g → 23255ms (shuffle_w=100KB, gc=1140ms)  # menor exec = mais lento
- iter 4: 4×12c/48g → **19387ms (best)** (shuffle_w=104KB, gc=4691ms)
best_config = 4×12c/48g, shuffle=100. Custo LLM ~US$0.001. (Configs fixadas em
ids-pb via node_selector; paralelismo real limitado à capacidade do nó — o sinal
de otimização entre configs é ilustrativo, não conclusivo nessa escala/pin.)
