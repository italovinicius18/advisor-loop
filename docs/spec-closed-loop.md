# advisor-loop — Fechamento do loop com execução real (SparkK8sBackend + OptimizationLoop)

**Data**: 2026-08-21
**Status**: rascunho para revisão do autor
**Repo alvo**: `advisor-loop` (o código; a spec e o plano são trazidos para lá ao fim, como os anteriores)
**Estende**: o harness advisor-loop existente (proposal mode + probes reais + memória). Fecha o gap declarado no teste sf1-loop: as lições de memória hoje carregam "o que foi proposto", não "o que funcionou".

---

## 1. Objetivo

Fechar o ciclo do spec (Contexto → Sondagem → Recomendação → Validação → **Execução real** → Destilação → Memória) com execução Spark de verdade no cluster RNP. O advisor propõe uma config, o Spark roda a query com ela, a duração/status real volta como feedback, e o advisor **re-propõe** — iterando até convergir ou esgotar o budget. É a peça que transforma o advisor de "propositor de iteração-1" em "otimizador em loop com feedback empírico".

### Decisões de escopo (confirmadas pelo autor, 2026-08-21)
- **Submitter novo minimalista** (não reusar o harness `run_tpcds_queries_v3.py`): submit + driver PySpark próprios, isolados. **Reúso de assets**: SQL Spark das queries (`queries/*.sql`) e imagem Spark+Delta.
- **Parada**: máx N iterações (default 4) **OU** sem melhora de tempo por 2 iterações seguidas — o que vier primeiro.
- **Primeiro loop ao vivo**: SF1, 1–2 queries (barato/rápido; valida o loop fechado com baixo custo de cluster).

### Não-objetivos (YAGNI neste corte)
- `Supervisor` in-flight (kill adaptativo por métricas do driver) — para SF1 rápido, um timeout duro por job basta. Fica como complemento posterior (já construído, wiring depois).
- SF10+/regime pesado — depois de validar em SF1.
- Otimização multi-objetivo (custo × tempo) — objetivo é `min_time`.

---

## 2. Arquitetura

### 2.1 Componentes

| Componente | Responsabilidade | Interface |
|---|---|---|
| `SparkK8sBackend` | Executa 1 query no cluster com a config proposta; devolve status + duração + métricas de I/O reais | `execute(episode, rec) -> Outcome` (mesma interface de ReplayBackend/ProposalBackend; `QueryResult` ganha um campo `metrics: dict` com input/output/shuffle bytes, gc, throughput) |
| `SparkSubmitter` | Monta o launcher (config → spark-submit args), aplica via SSH→kubectl, espera, coleta o resultado | `submit(config, dataset, query, timeout_s) -> {status, duration_ms, error}` |
| `OptimizationLoop` | O laço: propõe → valida → executa → realimenta → re-propõe; para em N ou sem-melhora; rastreia a melhor config | `optimize(episode, deps, max_iters, patience) -> BestResult` |
| `Advisor.advise(..., history=[])` | (extensão) renderiza o histórico (config→resultado) desta query no prompt | history: `list[(ConfigVector, Outcome)]` |

### 2.2 ConfigVector → spark-submit (tradução)
```
executor_cores       → spark.executor.cores
executor_memory_gb   → spark.executor.memory = {N}g
executor_instances   → spark.executor.instances
driver_memory_gb     → spark.driver.memory = {N}g
shuffle_partitions   → spark.sql.shuffle.partitions
advisory_partition_mb→ spark.sql.adaptive.advisoryPartitionSizeInBytes = {N}m
aqe_enabled          → spark.sql.adaptive.enabled
broadcast_threshold_mb→ spark.sql.autoBroadcastJoinThreshold = {N}m (ou -1)
```
Confs fixas (do padrão exp05, `run_benchmark_v3.sh`): master `k8s://https://kubernetes.default.svc:443`, `spark.kubernetes.namespace=italo-unb`, `authenticate.driver.serviceAccountName=ns-admin`, imagem Spark+Delta, `spark.hadoop.fs.s3a.endpoint=http://minio:9000` (+ creds minioadmin, path-style, SSL off), extensões Delta (`io.delta.sql.DeltaSparkSessionExtension`, `DeltaCatalog`).

### 2.3 Driver PySpark minimalista (bundle na imagem dedicada)
Script enxuto, **empacotado na imagem** (§7.1), não via ConfigMap: (1) cria SparkSession com as confs recebidas; (2) registra as tabelas Silver Delta como temp views (`spark.read.format("delta").load("s3a://silver/{sf}/delta/parquet/{tabela}")`); (3) lê o SQL da query (`queries/{q}.sql`, também na imagem); (4) cronometra a execução; (5) **coleta métricas de I/O** via `QueryExecutionListener`/`SQLMetrics` do Spark: `input_bytes`, `output_bytes`, `shuffle_read_bytes`, `shuffle_write_bytes`, `gc_time_ms`, `duration_ms`, e deriva `read_throughput_mbps`/`write_throughput_mbps` (bytes/duração); (6) escreve **um JSON de resultado** com status + todas essas métricas em **dois lugares**: stdout (linha única, p/ os logs) **e** `s3a://advisor-results/{run_id}.json` no MinIO (robusto). Status: `success` | `failed` | `timeout` | `oom` (classificado do erro/exit). As métricas de I/O servem para validar melhorias além do tempo (throughput, volume lido/escrito) — pedido do autor.

### 2.4 Mecânica de submit (SSH→kubectl)
O advisor local não alcança a API do k8s; o submit roda **de dentro do cluster** (como o exp05). `SparkSubmitter`:
1. Renderiza um **launcher pod/Job** (imagem Spark) que executa `spark-submit` com as confs + o driver script + args (sf, query). Nome único por (episódio, iteração).
2. `kubectl apply` via SSH (`ssh … "kubectl apply -f -"` com o manifesto no stdin).
3. **Poll** do estado até Completed/Failed (ou timeout duro `max_wall_clock`), lendo `kubectl get pod` + logs.
4. Lê o resultado de `s3a://advisor-results/{run_id}.json` no MinIO (**primário**, robusto), com fallback para a linha JSON dos **logs** do driver. Ambos contêm status + métricas de I/O (§2.3).
5. Limpa o launcher/driver pods (`kubectl delete`).
Distinção de falha (lição do gotcha exp05): "Complete" em segundos = crash do driver → tratar como `failed`, não `success` (checar presença da linha JSON de resultado; ausência = crash).

### 2.5 O laço (`OptimizationLoop.optimize`)
```
history = []          # [(config, outcome)]
best = None           # (config, duration_ms)
no_improve = 0
para iter em 1..max_iters:
    ctx = pack.render(ep) + memórias + render_history(history)
    rec = advisor.advise(ep, ctx, probe, memórias, history=history)
    ok, motivo = guardian.validate(rec.config)
    se não ok: history.append((rec.config, Outcome(invalid_recommendation, motivo))); continue
    outcome = spark_backend.execute(ep, rec)         # RODA Spark
    journal(query_result, episode_iter) ; memory.distill(ep, rec, outcome)  # COM duração real
    history.append((rec.config, outcome))
    se outcome.status == success e (best is None ou dur < best.dur):
        best = (rec.config, dur); no_improve = 0
    senão: no_improve += 1
    se no_improve >= patience (default 2): break
retorna best  (ou "nenhuma config viável" se best is None)
```
`render_history`: linhas curtas tipo `"iter 2: 8c/28g/8w → 43.1s"` / `"iter 3: 24c/96g/6w → OOM"`, para o advisor raciocinar sobre o que melhorou/piorou.

---

## 3. Feedback no Advisor
`advise()` ganha `history: list[tuple[ConfigVector, Outcome]] = []`. Quando não-vazio, injeta no contexto (antes do submit_recommendation): "Resultados anteriores nesta query (min_time): {render_history}. Proponha uma config que reduza o tempo; se uma falhou por OOM, ajuste memória/paralelismo." Mantém as instruções existentes (concisão, tabelas reais, TRY_CAST). A tool `submit_recommendation` é a mesma.

---

## 4. Erro e segurança
- **Guardian antes do submit** (já existe): config inviável não gasta cluster.
- **Timeout duro por job** (`max_wall_clock`, default 20 min p/ SF1) no `SparkSubmitter` — mata o driver via `kubectl delete` e retorna `timeout`.
- **1 job por vez** por design (lição de contaminação do spec) — o loop é sequencial; nada de submits paralelos no mesmo cluster.
- **OOM/timeout viram feedback**, não crash: o outcome com status alimenta a próxima iteração.
- **Limpeza garantida**: pods do launcher/driver deletados ao fim de cada iteração (finally), mesmo em erro.
- **Segredos**: creds MinIO já são defaults conhecidos (minioadmin) no cluster; nada novo exposto. A chave OpenRouter não toca o cluster (LLM roda local).

---

## 5. Testes
- `SparkSubmitter`: unit test da tradução `config → spark-submit args/manifesto` (string do manifesto contém `spark.executor.instances=8` etc.) e do **parse do resultado** (linha JSON → Outcome; ausência de linha → `failed`/driver_crash) com `subprocess`/SSH **mockado**. Nenhum cluster nos unit tests.
- `OptimizationLoop`: unit test do laço com um **backend fake** que devolve durações roteirizadas — verifica: para em `patience`, para em `max_iters`, escolhe a melhor, trata invalid/failed sem crash, threading do history.
- `advise(history=...)`: unit test de que o history entra no prompt (FakeClient inspeciona as mensagens).
- **Validação ao vivo (controller)**: 1 loop fechado real em SF1, 1 query — confirma submit→execução→feedback→re-proposta end-to-end e mede custo (cluster + API).

---

## 6. Marcos
| Fase | Entrega |
|---|---|
| F1 | Driver PySpark minimalista + `SparkSubmitter` (tradução + manifesto + parse), unit-tested com SSH/subprocess mock |
| F2 | `SparkK8sBackend` (usa SparkSubmitter) + `OptimizationLoop` + `advise(history)` + wiring (campanha `backend: spark_k8s`), unit-tested com fakes |
| F3 | Imagem `spark-advisor:3.5.0-delta` construída (driver + queries bundled) e publicada no GHCR + 1 loop fechado ao vivo SF1/q03; medir custo (cluster + API) e as métricas de I/O |

---

## 7. Decisões tomadas (autor, 2026-08-21)
1. **Imagem Spark dedicada ao projeto** (não reusar a `spark-benchmark` do exp05): construir uma nova, `ghcr.io/italovinicius18/spark-advisor:3.5.0-delta`, com Spark 3.5 + Delta + o **driver PySpark** e os **`queries/*.sql`** empacotados (Dockerfile próprio em `service/` ou `docker/spark-advisor/`). Pública no GHCR (como o probe), cluster puxa sem secret.
2. **Resultado + métricas via AMBOS**: o driver escreve o JSON (status + métricas de I/O) no **MinIO** (`s3a://advisor-results/{run_id}.json`, primário) **e** no **stdout/logs** (fallback). Métricas de I/O (input/output/shuffle bytes, gc, throughput leitura/escrita) capturadas junto com a duração — úteis para validar melhorias além do tempo.
3. **Query do primeiro loop**: `q03` (tanto faz; o autor testará todas depois).

## 8. Riscos
- **Latência por iteração**: cada iteração = 1 submit + fila + execução Spark (minutos mesmo em SF1). Loop de 4 iterações × poucas queries = dezenas de min de cluster. Mitigado por SF1 + parada por sem-melhora.
- **Imagem/fila do cluster**: se o cluster estiver ocupado, o submit espera. Mitigado por 1-por-vez e timeout.
- **Driver crash mascarado**: tratado pela checagem da linha JSON (§2.4).
- **LLM propõe sempre a mesma config** (temp 0.2, vimos convergência): o history explícito ("já testei X→43s") deve empurrar variação; se estagnar, a parada por sem-melhora encerra cedo (barato).
