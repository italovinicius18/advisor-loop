# advisor-loop — Harness de Episódios de Recomendação de Configuração via LLM

**Data**: 2026-08-19
**Status**: rascunho para revisão do autor
**Repo alvo**: repositório privado dedicado `advisor-loop` (a criar após aprovação; este spec vive em branch própria até lá)
**Natureza**: trabalho autocontido — este documento não depende de, nem referencia, experimentos ou resultados anteriores

---

## 1. Visão

Sistemas de processamento de dados distribuídos (ex.: Spark sobre Kubernetes)
expõem dezenas de parâmetros de configuração cuja escolha determina se uma
carga SQL analítica executa bem, executa mal ou simplesmente não executa.
Encontrar a configuração certa hoje é artesanal: tentativa e erro, caro em
tempo de cluster e não sistematizado.

O **advisor-loop** trata cada decisão de configuração como um **episódio**
reprodutível e auditável, conduzido por um LLM com acesso a contexto técnico,
ferramentas de sondagem de dados e memória de episódios anteriores:

```
Episódio = (workload, dataset, query[s], objetivo)
   → Contexto    (knowledge pack versionado)
   → Sondagem    (tools de leitura leve sobre as tabelas, com budget imposto)
   → Recomendação (LLM emite vetor de configuração validável)
   → Validação   (Guardian: quotas, restrições, preflight de infraestrutura)
   → Execução    (backend plugável, com supervisão in-flight)
   → Destilação  (resultado + lição em formato estruturado)
   → Memória     (persistida; episódios futuros recuperam)
```

O mecanismo de aprendizagem é **in-context learning com memória episódica**
(não é reinforcement learning: não há política nem gradiente — a nomenclatura
importa para publicação).

### 1.1 Vocabulário de granularidade

| Conceito | O que é | Unidade de quê | Budget associado | Exemplo |
|---|---|---|---|---|
| **Task** | 1 query × 1 config × 1 try | Retomada/idempotência (chave do Journal) | — | `(q17, config_hash=ab12, try=2)` |
| **Tentativa** | 1 configuração proposta + sua execução, dentro de um episódio | Iteração da busca de config | Conta contra o máx. de tentativas do episódio | "tentativa 3/5: Advisor subiu shuffle_partitions após OOM" |
| **Episódio** | 1 problema de aconselhamento `(workload, dataset, query[s], objetivo)` resolvido do início ao fim | **Unidade científica** — o data point das RQs (regret, viabilidade, falso kill) | Máx. tentativas (default 5) + hard cap de horas | "recomende config para q17 em escala 100" |
| **Campanha** | Conjunto de episódios sob protocolo fixo, declarado em YAML (nível L0–L2, provedor, split, budgets) | **Unidade operacional** — lock de cluster, teto de horas/custo, publicação do Archiver | Teto de horas de cluster + teto de $ de API | `auditoria-L1-sonnet-loso.yaml` |
| **Estudo** | Conjunto de campanhas que responde uma RQ | Unidade de análise/escrita | Orçamento total do projeto | RQ-A = 3 níveis × 3 provedores = 9 campanhas |

Um episódio de *auditoria* (RQ-A/B/C) tem tipicamente 1 tentativa (recomendou →
executou → mediu regret). Um episódio de *viabilidade* (RQ-D) tem até 5
tentativas (loop até executar ou esgotar). O critério de parada de cada
campanha vive no YAML dela — parar é decisão de protocolo, não de cansaço.

### Não-objetivos (YAGNI)

- Não é um auto-tuner de produção; é instrumento de pesquisa.
- Não otimiza ingestão/ETL — o alvo é a execução de queries analíticas.
- Não faz fine-tuning de modelos (trabalho futuro).
- Não suporta campanhas concorrentes no mesmo cluster (1 execução por vez,
  por design, para não contaminar medições).

---

## 2. Perguntas de pesquisa servidas

- **RQ-A (qualidade do conselho)**: quão boas são as configurações
  recomendadas por LLMs, medidas contra referências empíricas (quando um
  grid de referência medido existir) e contra baselines (config default,
  heurísticas de dimensionamento)?
- **RQ-B (sondagem)**: acesso a metadados e estatísticas reais das tabelas
  (probes) melhora a recomendação vs. apenas descrição textual da carga?
- **RQ-C (memória)**: experiência destilada de episódios anteriores melhora
  recomendações futuras (ablation memória ON/OFF, com higiene de splits)?
- **RQ-D (fronteira de viabilidade)**: em cargas/escalas onde execuções
  falham ou nunca foram tentadas, o advisor torna a carga executável — e em
  quantas iterações? Resultado negativo também responde: o advisor reconhece
  inviabilidade ou insiste em configurações fadadas?
- **RQ-E (supervisão in-flight)**: um LLM observando métricas do job em
  execução decide melhor que um TTL estático quando continuar vs.
  abortar-e-reconfigurar? Métricas: horas de cluster desperdiçadas, falsos
  kills, tempo até primeiro sucesso numa busca de configuração.

---

## 3. Arquitetura

### 3.1 Componentes (cada um: um propósito, uma interface, testável isolado)

| Componente | Responsabilidade | Interface (resumo) |
|---|---|---|
| `KnowledgePack` | Contexto técnico versionado: espaço de knobs do engine (docs oficiais, versão pinada), specs do cluster alvo (nodes, quotas, storage), restrições declaradas | `pack.render(episode) -> str`; arquivos YAML/MD com hash de versão |
| `ProbeBackend` | Tools de sondagem sobre as tabelas da query. Implementação inicial: DuckDB lendo tabelas lakehouse (Delta) direto do object store — metadados e estatísticas do transaction log (zero scan) + amostragens limitadas | `list_tables(dataset)`, `table_stats(t)`, `sample_query(sql, budget)` |
| `ProbeBudget` | Impõe limites NO TOOL LAYER (não confia no LLM): máx. N probes, máx. bytes escaneados, timeout por probe, denylist (ex.: SELECT sem LIMIT) | encapsula `ProbeBackend`; excedeu → erro estruturado devolvido ao LLM |
| `Advisor` | Chama o LLM com contexto + tools; extrai vetor de configuração validável (JSON Schema) | `advise(episode, memory_hits) -> Recommendation`; adapter por provedor, temperatura fixa, N amostras configurável |
| `Guardian` | Valida a recomendação antes de executar: quotas do cluster, lista de configurações sabidamente inviáveis, preflight de infraestrutura (integridade do dataset via manifest, espaço em storage) | `validate(rec) -> ok \| rejected(reason)`; rejeição vira feedback estruturado ao Advisor (1 retry); 2ª rejeição encerra com `status=invalid_recommendation` (também é dado) |
| `ExecutionBackend` | Executa a configuração recomendada. Implementações: `ReplayBackend` (simulação por lookup num dataset de resultados de referência, quando disponível — permite desenvolver e avaliar o loop sem custo de cluster), `SparkK8sBackend` (spark-submit em Kubernetes), outros engines como adapters futuros | `execute(episode, rec) -> Outcome` |
| `Supervisor` | Loop de supervisão in-flight (backends vivos): coleta métricas do driver (API REST do engine: stages/tasks, shuffle, GC, spill, executors perdidos) + eventos do orquestrador (OOMKilled, restarts); em checkpoints, o LLM decide `continue \| kill(reason)` | `watch(job, policy) -> Decision*`; decisões e séries de métricas vão ao Journal |
| `Journal` | Verdade append-only: JSONL, uma linha por evento (probe, recomendação, decisão de supervisão, resultado por query), chave idempotente, sincronizado para fora do cluster a cada evento | `append(event)`, `has(task_key)`, `replay()` |
| `MemoryStore` | Destilações de episódios; recuperação por similaridade estruturada (workload, escala vizinha, classe de query), respeitando o split de avaliação | `distill(episode, outcome) -> Memory`, `retrieve(episode, split) -> [Memory]` |
| `CampaignRunner` | Orquestra episódios de uma campanha declarada em YAML; retomada exata via Journal; lock global anti-concorrência | `run(campaign.yaml)`; CLI |
| `Archiver` | Política de evidência: ao fim de cada episódio/campanha, empacota e publica os artefatos no repositório Git (ver §5) | `publish(campaign) -> commit` |

### 3.2 Vetor de configuração (action space, fechado e validável)

```json
{
  "executor_cores": int,
  "executor_memory_gb": int,
  "executor_instances": int,
  "driver_memory_gb": int,
  "shuffle_partitions": int,
  "advisory_partition_mb": int,
  "aqe_enabled": bool,
  "broadcast_threshold_mb": int
}
```

Fechado por design: (a) validável pelo Guardian contra os limites declarados
no KnowledgePack do cluster alvo; (b) comparável entre modelos e episódios;
(c) extensível por engine via schema versionado (um `action_schema.json` por
backend).

### 3.3 Ciclo do episódio (data flow)

1. `CampaignRunner` lê o próximo episódio pendente (o Journal decide o que
   falta — retomada é o caso normal, não a exceção).
2. `KnowledgePack.render()` monta o contexto (hash do pack → Journal).
3. `MemoryStore.retrieve()` traz até K destilações permitidas pelo split
   (modo memória-OFF pula este passo).
4. `Advisor.advise()` roda o loop de tool-calling com
   `ProbeBudget(ProbeBackend)`; toda chamada de tool vira evento no Journal.
5. `Guardian.validate()`; rejeição → 1 retry com o motivo.
6. `ExecutionBackend.execute()`; por query: resultado → Journal (flush+sync).
   6b. Em backends vivos, o `Supervisor` roda em paralelo (§3.4); `kill`
   encerra o episódio como `aborted_by_supervisor` e realimenta o Advisor.
7. `MemoryStore.distill()` grava a lição (modo memória-ON).
8. `Archiver.publish()` versiona os artefatos no Git.

### 3.4 Supervisor — continuar/abortar sem TTL estático

Substitui timeouts fixos por decisão adaptativa, com backstops
não-negociáveis (um LLM "otimista" não pode queimar cluster indefinidamente):

- **Cadência**: métricas a cada 60s; checkpoint de DECISÃO LLM a cada N min
  (default 10) ou em gatilhos (executor morto, stage sem progresso por M min,
  GC acima de limiar). Entre checkpoints, nenhuma chamada de API.
- **Janela mínima de observação** (default 10 min): nenhum kill antes dela —
  evita matar warm-up.
- **Hard cap externo** (`max_wall_clock`, default 8h, por campanha): backstop
  de orçamento de cluster, não mecanismo de qualidade. O LLM decide DENTRO
  desse envelope.
- **Decisão estruturada**: `{action: continue|kill, reason, evidence:
  [métricas citadas], confidence}` — validável, logada, destilável em memória.
- **Avaliação (RQ-E)**: contra baselines de TTL fixo, nas métricas (a) horas
  desperdiçadas, (b) falsos kills, (c) tempo até primeiro sucesso.

### 3.5 Modos de avaliação (ablation ladder)

| Nível | Contexto | Probes | Memória | Custo |
|---|---|---|---|---|
| L0 | estático (pack) | não | não | só API |
| L1 | pack | sim | não | API + probes |
| L2 | pack | sim | sim | API + probes |

Baselines não-LLM: configuração default do engine; heurística simples de
dimensionamento; e, quando existir grid de referência medido, o ótimo
empírico (oráculo). Todos os níveis × N provedores × splits.

---

## 4. Protocolo anti-vazamento

- **Splits**: validação leave-one-out por escala e por query. A memória
  recuperável num episódio de avaliação NUNCA contém destilações do próprio
  (escala, query) avaliado — filtro em `retrieve(split)`.
- **KnowledgePack** contém apenas conhecimento público (docs do engine) +
  specs declaradas do cluster; experiência empírica entra exclusivamente via
  MemoryStore, controlado por split.
- **Ameaças declaradas**: benchmarks SQL públicos podem estar no treinamento
  dos LLMs — o estudo mede qualidade de conselho, não generalização a
  workload secreto; não-determinismo mitigado com temperatura fixa +
  N amostras por episódio.

---

## 5. Política de evidência — nada autoritativo fica no cluster

Requisito de primeira classe: **todo artefato gerado é versionado no GitHub**
e utilizável diretamente na escrita da pesquisa.

- O `Journal` é sincronizado para fora do cluster **a cada evento** (nunca
  apenas ao final do run); o cluster é descartável a qualquer momento.
- Ao fim de cada episódio/campanha, o `Archiver` commita e push:
  `evidence/<campanha>/<episódio>/` com journal (JSONL comprimido),
  recomendações, decisões do Supervisor, destilações de memória, hash do
  KnowledgePack, versões de modelo/prompt e sumários prontos para o texto
  (tabelas CSV/Markdown).
- Arquivos grandes (>50 MB) entram comprimidos; acima do limite do GitHub,
  via Git LFS — decisão registrada por campanha.
- Nenhum segredo no repo: credenciais via ambiente; o `Archiver` valida
  (scan de padrões) antes do push.

## 6. Confiabilidade — requisitos de resiliência

| Modo de falha previsto | Requisito |
|---|---|
| Dataset corrompido/alterado no storage entre execuções | **Preflight de integridade**: manifest dos arquivos do transaction log verificado contra o object store antes de cada campanha e a cada retomada; divergência → aborta antes de gastar cluster |
| Perda de resultados por falha de pod/serviço | **Journal-first**: evento persistido e sincronizado a cada query (local + object store + Git), não no fim do run |
| Falha no meio de uma campanha longa | **Idempotência por task key** `(workload, dataset, query, config_hash, try)`: retomada pula o que o Journal já tem; granularidade de retomada = 1 query — nunca se re-executa uma escala inteira |
| Job "completa" em segundos mascarando crash do driver | `Outcome` distingue `driver_crash` de `completed` via duração mínima plausível + validação da contagem de resultados |
| Interferência de execuções concorrentes | Lock global (lease no orquestrador) antes de qualquer backend vivo; recusa-se a rodar se houver outra carga ativa no namespace |
| Estouro de recursos do cluster | Guardian valida contra quota ANTES do submit; Supervisor + hard cap por episódio |

---

## 7. Estrutura do repo (`advisor-loop`, privado)

```
advisor-loop/
  pyproject.toml
  advisor_loop/
    episode.py          # dataclasses: Episode, Recommendation, Outcome, Decision
    knowledge/          # KnowledgePack + packs/ (ex.: spark-3.5.yaml, cluster.yaml)
    probes/             # ProbeBackend (duckdb_delta.py), ProbeBudget
    advisor/            # Advisor + providers/
    guardian.py
    backends/           # replay.py, spark_k8s.py
    supervisor.py
    journal.py
    memory/             # MemoryStore, distill.py, retrieval.py
    archiver.py
    runner.py           # CampaignRunner + CLI
  campaigns/            # YAMLs declarativos
  evidence/             # artefatos publicados pelo Archiver (§5)
  tests/
```

## 8. Marcos (alvo de submissão: novembro/2026)

| Fase | Entrega |
|---|---|
| F1 (2 semanas) | Núcleo: episode/journal/replay backend + probe DuckDB local + Advisor L0 + Guardian; campanha piloto pequena decide o protocolo |
| F2 (2 semanas) | L1 (probes com budget) + L2 (memória + splits); campanhas de auditoria em replay; Archiver operante desde o 1º episódio |
| F3 (2 semanas, condicional a cluster) | SparkK8sBackend + Supervisor; campanhas vivas de fronteira de viabilidade (RQ-D) em escalas crescentes, incluindo escala inédita |
| F4 (3–4 semanas) | Análise + escrita + artefato reproduzível público |

As fases vivas (F3) são condicionais: o estudo mínimo publicável sai de
F1+F2. Go/no-go explícito ao fim de F2.

## 9. Decisões tomadas (2026-08-19, confirmadas pelo autor)

1. **Nome do repo**: `advisor-loop`. Renomear para um "nome de sistema" só se
   o paper pedir, na véspera da submissão.
2. **Provedores LLM — desenho em dois tiers** (versões pinadas e logadas no
   Journal). Custo por episódio L1 (≈ 45k tokens de entrada com caching ≈ 15k
   efetivos + ~4k de saída):
   - **Grid completo** (todos os episódios, 1 amostra nas campanhas
     exploratórias): DeepSeek V4 Flash 0731 ($0,0765/$0,153 por MTok —
     ~US$0,002/ep) e Qwen3.7 Flash ($0,03/$0,13 — ~US$0,001/ep), ambos via
     OpenRouter (conta criada 2026-08-19; chave em `.env`, NUNCA versionada —
     protegida por `.gitignore` + `info/exclude`). Dois open-weights de labs
     independentes cobrem a variação entre modelos no fatorial inteiro por
     ~US$15 (grid duplo completo, 3 amostras). GLM 5.2 e Kimi K2.6 ficam
     anotados como alternativos caso um dos dois falhe no piloto (ex.:
     tool-calling instável); rotas free (Nemotron, North Mini) só em
     exploração — rate limits inviabilizam campanha.
   - **Subamostra estratificada frontier** (10–15% dos episódios, escolhidos
     por estrato de escala × classe de query × nível, 3 amostras): Claude
     Sonnet 5 ($3/$15; intro $2/$10 até 31/08 — ~US$0,10/ep) ≈ US$40–60.
     Imuniza contra a crítica "só testaram modelos baratos, um frontier já
     resolveria" e mede se pagar 30–100× por token compra configs melhores —
     achado nos dois desfechos. Segundo frontier descartado: não adiciona
     valor de claim proporcional ao custo.
   - **Supervisor**: gpt-oss-20b ($0,03/$0,13; 251ms de latência, 90 t/s —
     perfil ideal para muitas decisões pequenas) via OpenRouter; Haiku 4.5
     na subamostra frontier. Custo marginal (<US$2 no estudo inteiro).
   - **Custo real logado**: o Journal registra o custo efetivo de CADA
     chamada (a API do OpenRouter retorna custo por generation) — preços de
     tabela/promoções flutuam; a contabilidade do estudo usa o medido, não o
     anunciado.
   **Teto total revisado: ~US$60–80** (era US$700–800 na 1ª estimativa). Os
   US$10 de crédito inicial cobrem o piloto e os dois grids open-weights em
   modo exploratório (1 amostra ≈ US$5); recarga só após o piloto validar.
3. **Memória**: retrieval estruturado simples (workload / escala vizinha /
   classe de query, K=5). **Embeddings ficam mapeados como extensão futura**:
   a interface `MemoryStore.retrieve()` é agnóstica ao mecanismo — trocar o
   retrieval não toca o resto do sistema; promover a embeddings apenas se o
   piloto mostrar misses reais de recuperação (e isso vira achado reportável).
4. **Cadência do Supervisor**: gatilhos primeiro (executor morto, stage
   estagnado, GC acima do limiar → avaliação imediata), relógio de 10 min
   como fallback; heurística barata filtra o caso "tudo normal" sem chamar o
   LLM. Janela mínima 10 min, hard cap 8h.
5. **Granularidade e budgets**: conforme §1.1. Episódio de viabilidade: máx.
   5 tentativas. Campanha viva: teto de 40h de cluster. **Escala inicial das
   campanhas vivas: SF100** — escalas menores não entram (pouco interesse
   científico); a validação barata do loop acontece no `ReplayBackend`, que
   custa zero cluster, então o backend vivo já estreia em SF100 e sobe a
   escada a partir daí.
6. **Artefatos comprimidos (zstd), sem Git LFS**: journals particionados por
   episódio ficam em poucos MB; LFS só seria reavaliado se um artefato único
   inevitavelmente passasse de 100 MB.

## 10. Riscos

- **Tempo (solo)**: mitigado pelo replay-first (estudo mínimo sem cluster) e
  go/no-go por fase.
- **Reviewer: "replay não é execução real"**: o replay usa medições reais
  como referência e a limitação é declarada; as fases vivas cobrem o regime
  real quando o cluster permitir.
- **Custo de API**: budget por episódio + cache de respostas; o Supervisor
  adiciona chamadas recorrentes (mitigado pela cadência e gatilhos).
- **Escalas inéditas podem ser inviáveis no cluster alvo**: resultado
  negativo também responde RQ-D — o episódio é publicável nos dois desfechos.
- **Falso kill do Supervisor**: risco medido, não só mitigado — é a métrica
  (b) do RQ-E; janela mínima + evidência obrigatória reduzem a taxa.
