# advisor-loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir o harness `advisor-loop` — episódios auditáveis de recomendação de configuração via LLM, com journal idempotente, probes DuckDB com budget, guardian, replay backend, memória episódica, supervisor e archiver Git-first.

**Architecture:** Pipeline de episódio (Contexto → Sondagem → Recomendação → Validação → Execução → Destilação → Memória) com componentes plugáveis atrás de interfaces pequenas. Fase 1 roda 100% offline via `ReplayBackend`; backends vivos entram depois sem tocar o resto. Verdade append-only no `Journal`; artefatos publicados no Git pelo `Archiver`.

**Tech Stack:** Python ≥3.11, pydantic v2, httpx (OpenRouter — único endpoint para todos os modelos), duckdb, pyarrow, pyyaml, zstandard, python-dotenv, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-advisor-loop-design.md` (branch `advisor-loop` do repo da dissertação; copiar para `docs/spec.md` do repo novo no Task 0)

## Global Constraints

- Repo novo **privado** `advisor-loop`; bootstrap inicial na `main` (uma vez), depois todo trabalho em branches `feat/*`.
- Segredos SÓ via ambiente: `OPENROUTER_API_KEY` lida de `.env` (python-dotenv); `.env` no `.gitignore` desde o primeiro commit.
- Modelos pinados (strings exatas, sempre logadas no Journal): grid `deepseek/deepseek-v4-flash-0731` e `qwen/qwen3.7-flash`; frontier `anthropic/claude-sonnet-5`; supervisor `openai/gpt-oss-20b` — todos via OpenRouter.
- `temperature=0.2` fixa em toda chamada; custo real por chamada capturado da resposta do OpenRouter e gravado no Journal.
- `task_key = sha256("{workload}|{dataset}|{query}|{config_hash}|{try_n}")[:16]`; toda execução é idempotente por task_key.
- ConfigVector fechado (bounds do spec §3.2); validação SEMPRE via Guardian antes de qualquer execução.
- Journal é append-only JSONL; nenhum componente sobrescreve eventos.
- Python ≥3.11; testes com pytest; TDD em todo task (teste falha → implementa → passa → commit).

---

### Task 0: Bootstrap do repositório privado

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `advisor_loop/__init__.py`, `tests/__init__.py`, `docs/spec.md` (cópia do spec)

**Interfaces:**
- Produces: repo `advisor-loop` privado no GitHub do usuário; pacote `advisor_loop` instalável em modo dev; `pytest` verde (0 testes).

- [ ] **Step 1: Criar o repo privado e o diretório local**

```bash
gh repo create advisor-loop --private --description "LLM configuration-advisor episode harness" --clone
cd advisor-loop
```

- [ ] **Step 2: Escrever pyproject.toml e .gitignore**

```toml
# pyproject.toml
[project]
name = "advisor-loop"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.7", "httpx>=0.27", "duckdb>=1.0", "pyarrow>=16",
  "pyyaml>=6", "zstandard>=0.22", "python-dotenv>=1.0",
]
[project.optional-dependencies]
dev = ["pytest>=8"]
[project.scripts]
advisor-loop = "advisor_loop.runner:main"
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

```gitignore
# .gitignore
.env
*.env
!.env.example
__pycache__/
*.egg-info/
.venv/
data/*.parquet
```

- [ ] **Step 3: Copiar o spec, criar pacote vazio e instalar**

```bash
cp <dissertação>/docs/superpowers/specs/2026-08-19-advisor-loop-design.md docs/spec.md
mkdir -p advisor_loop tests && touch advisor_loop/__init__.py tests/__init__.py
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest  # esperado: "no tests ran"
```

- [ ] **Step 4: Commit de bootstrap (única push direto na main) e criar branch de trabalho**

```bash
git add -A && git commit -m "chore: bootstrap advisor-loop (pyproject, gitignore, spec)"
git push -u origin main
git switch -c feat/core
```

---

### Task 1: Modelos de dados (`episode.py`)

**Files:**
- Create: `advisor_loop/episode.py`
- Test: `tests/test_episode.py`

**Interfaces:**
- Produces: `ConfigVector` (8 campos do spec §3.2, com bounds), `Episode(workload, dataset, queries, objective, split, level, provider)`, `Recommendation(config: ConfigVector, rationale: str, samples: int)`, `Outcome(status, per_query: dict[str, QueryResult], wall_clock_s, cost_usd)`, `QueryResult(status, duration_ms, error)`, `Decision(action, reason, evidence, confidence)`, e `task_key(workload, dataset, query, config_hash, try_n) -> str` + `ConfigVector.config_hash() -> str` (sha256 dos campos ordenados, 16 hex).

- [ ] **Step 1: Teste falhando**

```python
# tests/test_episode.py
from advisor_loop.episode import ConfigVector, task_key
import pydantic, pytest

def test_config_vector_bounds():
    cv = ConfigVector(executor_cores=4, executor_memory_gb=14, executor_instances=4,
                      driver_memory_gb=8, shuffle_partitions=400,
                      advisory_partition_mb=64, aqe_enabled=True, broadcast_threshold_mb=10)
    assert cv.executor_cores == 4
    with pytest.raises(pydantic.ValidationError):
        ConfigVector(executor_cores=99, executor_memory_gb=14, executor_instances=4,
                     driver_memory_gb=8, shuffle_partitions=400,
                     advisory_partition_mb=64, aqe_enabled=True, broadcast_threshold_mb=10)

def test_config_hash_stable_and_task_key():
    kw = dict(executor_cores=4, executor_memory_gb=14, executor_instances=4,
              driver_memory_gb=8, shuffle_partitions=400, advisory_partition_mb=64,
              aqe_enabled=True, broadcast_threshold_mb=10)
    assert ConfigVector(**kw).config_hash() == ConfigVector(**kw).config_hash()
    k = task_key("w", "d100", "q17", ConfigVector(**kw).config_hash(), 2)
    assert len(k) == 16
```

- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_episode.py -v` → FAIL (módulo inexistente)

- [ ] **Step 3: Implementar**

```python
# advisor_loop/episode.py
import hashlib, json
from typing import Literal
from pydantic import BaseModel, Field

class ConfigVector(BaseModel):
    executor_cores: int = Field(ge=1, le=24)
    executor_memory_gb: int = Field(ge=4, le=96)
    executor_instances: int = Field(ge=1, le=20)
    driver_memory_gb: int = Field(ge=4, le=16)
    shuffle_partitions: int = Field(ge=100, le=4000)
    advisory_partition_mb: int = Field(ge=16, le=256)
    aqe_enabled: bool
    broadcast_threshold_mb: int = Field(ge=-1, le=512)

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

def task_key(workload: str, dataset: str, query: str, config_hash: str, try_n: int) -> str:
    raw = f"{workload}|{dataset}|{query}|{config_hash}|{try_n}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

class QueryResult(BaseModel):
    status: Literal["success", "failed", "timeout", "killed"]
    duration_ms: float | None = None
    error: str | None = None

class Outcome(BaseModel):
    status: Literal["completed", "driver_crash", "aborted_by_supervisor",
                    "invalid_recommendation", "infeasible", "error"]
    per_query: dict[str, QueryResult] = {}
    wall_clock_s: float = 0.0
    cost_usd: float = 0.0

class Recommendation(BaseModel):
    config: ConfigVector
    rationale: str
    samples: int = 1

class Decision(BaseModel):
    action: Literal["continue", "kill"]
    reason: str
    evidence: list[str]
    confidence: float = Field(ge=0, le=1)

class Episode(BaseModel):
    workload: str
    dataset: str            # ex.: "sf100"
    queries: list[str]
    objective: Literal["min_time", "feasibility"] = "min_time"
    split: str = "none"     # ex.: "loso-scale:sf100"
    level: Literal["L0", "L1", "L2"] = "L0"
    provider: str           # model string pinada
```

- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_episode.py -v` → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: episode data models (ConfigVector, Outcome, task_key)"`

---

### Task 2: Journal append-only idempotente (`journal.py`)

**Files:**
- Create: `advisor_loop/journal.py`
- Test: `tests/test_journal.py`

**Interfaces:**
- Consumes: `task_key` (Task 1).
- Produces: `Journal(path)` com `append(event: dict) -> None` (adiciona `ts` ISO-UTC; flush+fsync), `has(task_key: str) -> bool`, `events(kind: str | None = None) -> list[dict]`. Convenção: todo evento tem `kind` (`probe|recommendation|decision|query_result|episode_end|llm_call`) e opcionalmente `task_key`.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_journal.py
from advisor_loop.journal import Journal

def test_append_has_and_replay(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.append({"kind": "query_result", "task_key": "abc123", "status": "success"})
    assert j.has("abc123") and not j.has("zzz")
    j2 = Journal(tmp_path / "j.jsonl")          # reabre: índice reconstruído do disco
    assert j2.has("abc123")
    assert j2.events("query_result")[0]["status"] == "success"
    assert "ts" in j2.events()[0]
```

- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_journal.py -v` → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/journal.py
import json, os
from datetime import datetime, timezone
from pathlib import Path

class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: set[str] = set()
        self._events: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                self._events.append(ev)
                if ev.get("task_key"):
                    self._keys.add(ev["task_key"])

    def append(self, event: dict) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with open(self.path, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._events.append(event)
        if event.get("task_key"):
            self._keys.add(event["task_key"])

    def has(self, task_key: str) -> bool:
        return task_key in self._keys

    def events(self, kind: str | None = None) -> list[dict]:
        return [e for e in self._events if kind is None or e.get("kind") == kind]
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: append-only idempotent journal"`

---

### Task 3: KnowledgePack (`knowledge/`)

**Files:**
- Create: `advisor_loop/knowledge/__init__.py`, `advisor_loop/knowledge/pack.py`, `packs/spark-3.5.yaml`, `packs/cluster-example.yaml`
- Test: `tests/test_knowledge.py`

**Interfaces:**
- Produces: `KnowledgePack.load(pack_dir: Path) -> KnowledgePack` com `.render(episode: Episode) -> str` (contexto em texto), `.hash -> str` (sha256 do conteúdo dos YAMLs, 16 hex), `.cluster -> dict` (limites usados pelo Guardian: `max_cores_per_container`, `max_memory_gb_per_container`, `max_pods`, `total_cores`, `total_memory_gb`).

- [ ] **Step 1: Teste falhando**

```python
# tests/test_knowledge.py
from pathlib import Path
from advisor_loop.knowledge.pack import KnowledgePack
from advisor_loop.episode import Episode

def _write_packs(d: Path):
    (d / "engine.yaml").write_text(
        "engine: spark-3.5\nknobs:\n  shuffle_partitions: 'particoes de shuffle; sobe com volume'\n")
    (d / "cluster.yaml").write_text(
        "cluster:\n  max_cores_per_container: 24\n  max_memory_gb_per_container: 96\n"
        "  max_pods: 20\n  total_cores: 100\n  total_memory_gb: 400\n")

def test_load_render_hash(tmp_path):
    _write_packs(tmp_path)
    pack = KnowledgePack.load(tmp_path)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    ctx = pack.render(ep)
    assert "shuffle_partitions" in ctx and "sf100" in ctx
    assert len(pack.hash) == 16
    assert pack.cluster["max_pods"] == 20
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/knowledge/pack.py
import hashlib
from pathlib import Path
import yaml
from advisor_loop.episode import Episode

class KnowledgePack:
    def __init__(self, docs: dict[str, dict], hash_: str):
        self._docs = docs
        self.hash = hash_

    @classmethod
    def load(cls, pack_dir: str | Path) -> "KnowledgePack":
        pack_dir = Path(pack_dir)
        blobs, docs = [], {}
        for f in sorted(pack_dir.glob("*.yaml")):
            text = f.read_text()
            blobs.append(text)
            docs[f.stem] = yaml.safe_load(text)
        h = hashlib.sha256("".join(blobs).encode()).hexdigest()[:16]
        return cls(docs, h)

    @property
    def cluster(self) -> dict:
        for doc in self._docs.values():
            if "cluster" in doc:
                return doc["cluster"]
        raise KeyError("nenhum YAML do pack define 'cluster'")

    def render(self, episode: Episode) -> str:
        parts = [f"## Episodio\nworkload={episode.workload} dataset={episode.dataset} "
                 f"queries={','.join(episode.queries)} objetivo={episode.objective}"]
        for name, doc in self._docs.items():
            parts.append(f"## {name}\n{yaml.safe_dump(doc, allow_unicode=True)}")
        return "\n\n".join(parts)
```

Criar também os packs reais (conteúdo inicial mínimo, expandidos na campanha piloto):

```yaml
# packs/spark-3.5.yaml — knobs e impactos (resumo da doc oficial do Spark 3.5)
engine: spark-3.5
knobs:
  executor_cores: "cores por executor; paralelismo intra-executor"
  executor_memory_gb: "heap por executor; OOM se baixo p/ shuffle grande"
  executor_instances: "numero de executors (workers)"
  driver_memory_gb: "heap do driver; broadcasts e planos grandes pressionam"
  shuffle_partitions: "particoes de shuffle; sobe com volume de dados"
  advisory_partition_mb: "alvo do AQE p/ coalescing de particoes"
  aqe_enabled: "adaptive query execution; re-otimiza em runtime"
  broadcast_threshold_mb: "limite p/ broadcast join; -1 desliga (risco: piora joins pequenos)"
```

```yaml
# packs/cluster-example.yaml — template; o real e gerado por cluster alvo
cluster:
  max_cores_per_container: 24
  max_memory_gb_per_container: 96
  max_pods: 20
  total_cores: 100
  total_memory_gb: 400
  storage: "object store S3-compativel; tabelas Delta"
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: knowledge pack loader + initial packs"`

---

### Task 4: Probes DuckDB com budget (`probes/`)

**Files:**
- Create: `advisor_loop/probes/__init__.py`, `advisor_loop/probes/duckdb_probe.py`, `advisor_loop/probes/budget.py`
- Test: `tests/test_probes.py`

**Interfaces:**
- Produces: `DuckDBProbe(tables: dict[str, str])` (nome→path parquet/delta) com `list_tables() -> list[str]`, `table_stats(t) -> dict` (row_count, columns: [{name, type}] — via DuckDB, zero full-scan), `sample_query(sql: str) -> list[tuple]`; `BudgetedProbe(probe, max_probes=8, timeout_s=10)` que encapsula os três métodos, conta chamadas, rejeita SQL sem `LIMIT` e levanta `ProbeBudgetExceeded` além do máximo.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_probes.py
import duckdb, pytest
from advisor_loop.probes.duckdb_probe import DuckDBProbe
from advisor_loop.probes.budget import BudgetedProbe, ProbeBudgetExceeded

@pytest.fixture
def store(tmp_path):
    p = tmp_path / "t.parquet"
    duckdb.sql("SELECT range AS id, 'x' AS name FROM range(1000)").write_parquet(str(p))
    return {"store_sales": str(p)}

def test_stats_and_sample(store):
    probe = DuckDBProbe(store)
    assert probe.list_tables() == ["store_sales"]
    st = probe.table_stats("store_sales")
    assert st["row_count"] == 1000 and any(c["name"] == "id" for c in st["columns"])
    rows = probe.sample_query("SELECT id FROM store_sales LIMIT 5")
    assert len(rows) == 5

def test_budget_enforced(store):
    b = BudgetedProbe(DuckDBProbe(store), max_probes=2)
    b.table_stats("store_sales")
    with pytest.raises(ValueError):
        b.sample_query("SELECT * FROM store_sales")   # sem LIMIT -> rejeita
    b.sample_query("SELECT * FROM store_sales LIMIT 1")
    with pytest.raises(ProbeBudgetExceeded):
        b.list_tables()
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/probes/duckdb_probe.py
import duckdb

class DuckDBProbe:
    """Sondagem leve sobre tabelas parquet/delta. Paths locais ou s3:// (via httpfs/delta ext)."""
    def __init__(self, tables: dict[str, str]):
        self.tables = tables
        self.con = duckdb.connect()

    def _rel(self, table: str) -> str:
        path = self.tables[table]
        fn = "delta_scan" if not path.endswith(".parquet") else "read_parquet"
        return f"{fn}('{path}')"

    def list_tables(self) -> list[str]:
        return sorted(self.tables)

    def table_stats(self, table: str) -> dict:
        rc = self.con.sql(f"SELECT count(*) FROM {self._rel(table)}").fetchone()[0]
        cols = self.con.sql(f"DESCRIBE SELECT * FROM {self._rel(table)} LIMIT 0").fetchall()
        return {"row_count": rc, "columns": [{"name": c[0], "type": c[1]} for c in cols]}

    def sample_query(self, sql: str) -> list[tuple]:
        for name in self.tables:
            sql = sql.replace(name, self._rel(name))
        return self.con.sql(sql).fetchall()
```

```python
# advisor_loop/probes/budget.py
class ProbeBudgetExceeded(Exception):
    pass

class BudgetedProbe:
    def __init__(self, probe, max_probes: int = 8, timeout_s: int = 10):
        self._p, self.max_probes, self.used = probe, max_probes, 0
        self.timeout_s = timeout_s

    def _spend(self):
        if self.used >= self.max_probes:
            raise ProbeBudgetExceeded(f"budget de {self.max_probes} probes esgotado")
        self.used += 1

    def list_tables(self):
        self._spend()
        return self._p.list_tables()

    def table_stats(self, table: str):
        self._spend()
        return self._p.table_stats(table)

    def sample_query(self, sql: str):
        if "limit" not in sql.lower():
            raise ValueError("sample_query exige LIMIT explicito")
        self._spend()
        return self._p.sample_query(sql)
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: duckdb probe backend + enforced probe budget"`

---

### Task 5: Cliente LLM OpenRouter (`advisor/provider.py`)

**Files:**
- Create: `advisor_loop/advisor/__init__.py`, `advisor_loop/advisor/provider.py`
- Test: `tests/test_provider.py`

**Interfaces:**
- Produces: `OpenRouterClient(api_key: str | None = None, base_url="https://openrouter.ai/api/v1")` com `chat(model: str, messages: list[dict], tools: list[dict] | None = None, temperature=0.2) -> LLMResponse`; `LLMResponse(message: dict, tool_calls: list[dict], cost_usd: float, usage: dict)`. Formato OpenAI-compatible; `usage: {"include": true}` no payload para receber custo. Sem chave → lê `OPENROUTER_API_KEY` do ambiente (dotenv). Testes usam `httpx.MockTransport` — nunca rede real.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_provider.py
import httpx, json
from advisor_loop.advisor.provider import OpenRouterClient

def _mock(payload):
    def handler(request):
        body = json.loads(request.content)
        assert body["usage"] == {"include": True} and body["temperature"] == 0.2
        return httpx.Response(200, json=payload)
    return httpx.MockTransport(handler)

def test_chat_parses_cost_and_tool_calls():
    payload = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                 {"id": "c1", "function": {"name": "table_stats", "arguments": '{"table":"t"}'}}]}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.00042}}
    c = OpenRouterClient(api_key="k", transport=_mock(payload))
    r = c.chat("deepseek/deepseek-v4-flash-0731", [{"role": "user", "content": "oi"}])
    assert r.cost_usd == 0.00042
    assert r.tool_calls[0]["function"]["name"] == "table_stats"
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/advisor/provider.py
import os
import httpx
from dataclasses import dataclass, field
from dotenv import load_dotenv

@dataclass
class LLMResponse:
    message: dict
    tool_calls: list = field(default_factory=list)
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)

class OpenRouterClient:
    def __init__(self, api_key: str | None = None,
                 base_url: str = "https://openrouter.ai/api/v1",
                 transport: httpx.BaseTransport | None = None):
        load_dotenv()
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.http = httpx.Client(base_url=base_url, timeout=120,
                                 transport=transport,
                                 headers={"Authorization": f"Bearer {self.api_key}"})

    def chat(self, model: str, messages: list[dict],
             tools: list[dict] | None = None, temperature: float = 0.2) -> LLMResponse:
        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "usage": {"include": True}}
        if tools:
            payload["tools"] = tools
        resp = self.http.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        return LLMResponse(message=msg, tool_calls=msg.get("tool_calls") or [],
                           cost_usd=data.get("usage", {}).get("cost", 0.0),
                           usage=data.get("usage", {}))
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: openrouter client with real-cost capture"`

---

### Task 6: Advisor — loop de tools e recomendação (`advisor/advisor.py`)

**Files:**
- Create: `advisor_loop/advisor/advisor.py`
- Test: `tests/test_advisor.py`

**Interfaces:**
- Consumes: `OpenRouterClient.chat` (Task 5), `BudgetedProbe` (Task 4), `KnowledgePack.render` (Task 3), `Journal.append` (Task 2), modelos (Task 1).
- Produces: `Advisor(client, journal)` com `advise(episode, context: str, probe: BudgetedProbe | None, memories: list[str]) -> Recommendation`. Mecânica: tools OpenAI-format = probes (se `probe` dado — nível L1+) + tool obrigatória `submit_recommendation(config…, rationale)` cujo schema espelha `ConfigVector`; loop até o modelo chamar `submit_recommendation` (máx. 12 rounds → `AdvisorError`); toda chamada LLM vira evento `llm_call` no Journal (modelo, custo, tokens); probes viram eventos `probe`.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_advisor.py
import json, pytest
from advisor_loop.advisor.advisor import Advisor
from advisor_loop.advisor.provider import LLMResponse
from advisor_loop.episode import Episode
from advisor_loop.journal import Journal

CFG = {"executor_cores": 4, "executor_memory_gb": 14, "executor_instances": 4,
       "driver_memory_gb": 8, "shuffle_partitions": 400, "advisory_partition_mb": 64,
       "aqe_enabled": True, "broadcast_threshold_mb": 10}

class FakeClient:
    def __init__(self, scripted):
        self.scripted = list(scripted)
    def chat(self, model, messages, tools=None, temperature=0.2):
        return self.scripted.pop(0)

def _submit_call():
    args = json.dumps({**CFG, "rationale": "workload medio"})
    return LLMResponse(message={}, cost_usd=0.001, tool_calls=[
        {"id": "1", "function": {"name": "submit_recommendation", "arguments": args}}])

def test_advise_returns_recommendation_and_logs(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    adv = Advisor(FakeClient([_submit_call()]), j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    rec = adv.advise(ep, context="ctx", probe=None, memories=[])
    assert rec.config.executor_cores == 4
    assert j.events("llm_call")[0]["cost_usd"] == 0.001

def test_advise_gives_up_after_max_rounds(tmp_path):
    empty = LLMResponse(message={"content": "hmm"}, tool_calls=[])
    adv = Advisor(FakeClient([empty] * 12), Journal(tmp_path / "j.jsonl"))
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    with pytest.raises(Exception):
        adv.advise(ep, context="ctx", probe=None, memories=[])
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/advisor/advisor.py
import json
from advisor_loop.episode import ConfigVector, Episode, Recommendation
from advisor_loop.journal import Journal

class AdvisorError(Exception):
    pass

_SUBMIT = {
    "type": "function",
    "function": {
        "name": "submit_recommendation",
        "description": "Envia a configuracao final recomendada.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": [*ConfigVector.model_fields, "rationale"],
                       "properties": {**{k: {"type": "boolean" if k == "aqe_enabled" else "integer"}
                                         for k in ConfigVector.model_fields},
                                      "rationale": {"type": "string"}}}}}

_PROBE_TOOLS = [
    {"type": "function", "function": {"name": "list_tables",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "table_stats",
        "parameters": {"type": "object", "required": ["table"],
                       "properties": {"table": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "sample_query",
        "parameters": {"type": "object", "required": ["sql"],
                       "properties": {"sql": {"type": "string"}}}}},
]

class Advisor:
    def __init__(self, client, journal: Journal, max_rounds: int = 12):
        self.client, self.journal, self.max_rounds = client, journal, max_rounds

    def advise(self, ep: Episode, context: str, probe, memories: list[str]) -> Recommendation:
        tools = [_SUBMIT] + (_PROBE_TOOLS if probe else [])
        mem = "\n".join(f"- {m}" for m in memories) or "(sem memorias)"
        messages = [
            {"role": "system", "content":
             "Voce recomenda configuracao de execucao para cargas SQL analiticas. "
             "Use as tools de sondagem se disponiveis e finalize SEMPRE via submit_recommendation."},
            {"role": "user", "content": f"{context}\n\n## Memorias de episodios anteriores\n{mem}"},
        ]
        for _ in range(self.max_rounds):
            r = self.client.chat(ep.provider, messages, tools=tools)
            self.journal.append({"kind": "llm_call", "model": ep.provider,
                                 "cost_usd": r.cost_usd, "usage": r.usage})
            if not r.tool_calls:
                messages.append({"role": "assistant", "content": r.message.get("content") or ""})
                messages.append({"role": "user", "content":
                                 "Finalize chamando submit_recommendation."})
                continue
            messages.append({"role": "assistant", "content": None, "tool_calls": r.tool_calls})
            for call in r.tool_calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                if name == "submit_recommendation":
                    rationale = args.pop("rationale", "")
                    return Recommendation(config=ConfigVector(**args), rationale=rationale)
                result = getattr(probe, name)(**args)
                self.journal.append({"kind": "probe", "tool": name, "args": args})
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, default=str)[:4000]})
        raise AdvisorError(f"sem submit_recommendation apos {self.max_rounds} rounds")
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: advisor tool loop with forced recommendation submit"`

---

### Task 7: Guardian (`guardian.py`)

**Files:**
- Create: `advisor_loop/guardian.py`
- Test: `tests/test_guardian.py`

**Interfaces:**
- Consumes: `ConfigVector` (Task 1), `KnowledgePack.cluster` (Task 3).
- Produces: `Guardian(cluster: dict, infeasible: list[dict] = [])` com `validate(config: ConfigVector) -> tuple[bool, str]` — checa: cores/mem por container ≤ máximos; instances ≤ max_pods; total (cores, mem) ≤ totais do cluster; config ∉ lista de inviáveis (match por subconjunto de campos). Retorna `(False, motivo)` para feedback estruturado ao Advisor.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_guardian.py
from advisor_loop.guardian import Guardian
from advisor_loop.episode import ConfigVector

CLUSTER = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
           "max_pods": 20, "total_cores": 100, "total_memory_gb": 400}

def _cv(**over):
    base = dict(executor_cores=4, executor_memory_gb=14, executor_instances=4,
                driver_memory_gb=8, shuffle_partitions=400, advisory_partition_mb=64,
                aqe_enabled=True, broadcast_threshold_mb=10)
    return ConfigVector(**{**base, **over})

def test_ok_and_over_quota():
    g = Guardian(CLUSTER)
    assert g.validate(_cv())[0] is True
    ok, reason = g.validate(_cv(executor_cores=20, executor_instances=10))  # 200 > 100 cores
    assert ok is False and "total_cores" in reason

def test_known_infeasible():
    g = Guardian(CLUSTER, infeasible=[{"executor_memory_gb": 14, "executor_instances": 2}])
    ok, reason = g.validate(_cv(executor_instances=2))
    assert ok is False and "inviavel" in reason
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/guardian.py
from advisor_loop.episode import ConfigVector

class Guardian:
    def __init__(self, cluster: dict, infeasible: list[dict] | None = None):
        self.cluster = cluster
        self.infeasible = infeasible or []

    def validate(self, cfg: ConfigVector) -> tuple[bool, str]:
        c = self.cluster
        if cfg.executor_cores > c["max_cores_per_container"]:
            return False, f"executor_cores {cfg.executor_cores} > max_cores_per_container"
        if cfg.executor_memory_gb > c["max_memory_gb_per_container"]:
            return False, "executor_memory_gb acima do maximo por container"
        if cfg.executor_instances + 1 > c["max_pods"]:
            return False, "executors + driver excedem max_pods"
        if cfg.executor_cores * cfg.executor_instances > c["total_cores"]:
            return False, f"total_cores excedido ({cfg.executor_cores * cfg.executor_instances})"
        if cfg.executor_memory_gb * cfg.executor_instances + cfg.driver_memory_gb > c["total_memory_gb"]:
            return False, "memoria total excedida"
        d = cfg.model_dump()
        for known in self.infeasible:
            if all(d.get(k) == v for k, v in known.items()):
                return False, f"config sabidamente inviavel: {known}"
        return True, "ok"
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: guardian quota + known-infeasible validation"`

---

### Task 8: ReplayBackend + schema do dataset de referência (`backends/replay.py`)

**Files:**
- Create: `advisor_loop/backends/__init__.py`, `advisor_loop/backends/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `ConfigVector`, `Outcome`, `QueryResult` (Task 1).
- Produces: `ReplayBackend(reference: Path)` com `execute(episode, rec: Recommendation) -> Outcome` e `oracle(dataset, query) -> float | None` (melhor duration_ms medida). **Schema do parquet de referência** (contrato do Task 12): colunas `dataset (str), query (str), executor_cores (int), executor_memory_gb (int), executor_instances (int), status (str), duration_ms (float)`. Lookup: match exato nas 3 dims de recurso; sem match → `Outcome(status="infeasible")` com nota; knobs fora da matriz são registrados como `ignored_knobs` no Outcome... (campo extra via `per_query` não — usar journal do runner). Match parcial deliberadamente NÃO existe (limitação declarada no spec §3.5).

- [ ] **Step 1: Teste falhando**

```python
# tests/test_replay.py
import duckdb
from pathlib import Path
from advisor_loop.backends.replay import ReplayBackend
from advisor_loop.episode import Episode, Recommendation, ConfigVector

def _ref(tmp_path) -> Path:
    p = tmp_path / "ref.parquet"
    duckdb.sql("""SELECT * FROM (VALUES
        ('sf100','q1',4,14,4,'success',1200.0),
        ('sf100','q1',8,28,2,'success', 900.0),
        ('sf100','q2',4,14,4,'timeout', NULL))
        t(dataset,query,executor_cores,executor_memory_gb,executor_instances,status,duration_ms)
    """).write_parquet(str(p))
    return p

def _rec(cores, mem, inst):
    return Recommendation(rationale="", config=ConfigVector(
        executor_cores=cores, executor_memory_gb=mem, executor_instances=inst,
        driver_memory_gb=8, shuffle_partitions=400, advisory_partition_mb=64,
        aqe_enabled=True, broadcast_threshold_mb=10))

def test_execute_hit_miss_and_oracle(tmp_path):
    rb = ReplayBackend(_ref(tmp_path))
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    out = rb.execute(ep, _rec(4, 14, 4))
    assert out.per_query["q1"].duration_ms == 1200.0
    assert rb.execute(ep, _rec(12, 48, 6)).status == "infeasible"   # celula nao medida
    assert rb.oracle("sf100", "q1") == 900.0
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/backends/replay.py
from pathlib import Path
import duckdb
from advisor_loop.episode import Episode, Outcome, QueryResult, Recommendation

class ReplayBackend:
    def __init__(self, reference: str | Path):
        self.con = duckdb.connect()
        self.con.sql(f"CREATE VIEW ref AS SELECT * FROM read_parquet('{reference}')")

    def execute(self, ep: Episode, rec: Recommendation) -> Outcome:
        c = rec.config
        per_query, total = {}, 0.0
        for q in ep.queries:
            rows = self.con.execute(
                "SELECT status, duration_ms FROM ref WHERE dataset=? AND query=? "
                "AND executor_cores=? AND executor_memory_gb=? AND executor_instances=?",
                [ep.dataset, q, c.executor_cores, c.executor_memory_gb, c.executor_instances],
            ).fetchall()
            if not rows:
                return Outcome(status="infeasible", per_query={},
                               wall_clock_s=0.0, cost_usd=0.0)
            status, dur = rows[0]
            per_query[q] = QueryResult(status=status, duration_ms=dur)
            total += (dur or 0.0) / 1000
        return Outcome(status="completed", per_query=per_query, wall_clock_s=total)

    def oracle(self, dataset: str, query: str) -> float | None:
        row = self.con.execute(
            "SELECT min(duration_ms) FROM ref WHERE dataset=? AND query=? AND status='success'",
            [dataset, query]).fetchone()
        return row[0]
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: replay backend with oracle lookup"`

---

### Task 9: MemoryStore com higiene de split (`memory/store.py`)

**Files:**
- Create: `advisor_loop/memory/__init__.py`, `advisor_loop/memory/store.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `Episode`, `Outcome`, `Recommendation` (Task 1).
- Produces: `MemoryStore(path)` (JSONL) com `distill(episode, rec, outcome) -> dict` (grava `{workload, dataset, queries, config, status, lesson}` — lesson é string template determinística, sem LLM na v1) e `retrieve(episode, k=5) -> list[str]` que **exclui** memórias do mesmo `(dataset, query)` do episódio (anti-vazamento leave-one-out) e prioriza mesmo workload + dataset vizinho.

- [ ] **Step 1: Teste falhando**

```python
# tests/test_memory.py
from advisor_loop.memory.store import MemoryStore
from advisor_loop.episode import Episode, Outcome, QueryResult, Recommendation, ConfigVector

def _ep(dataset, query):
    return Episode(workload="w", dataset=dataset, queries=[query], provider="m")

def _rec():
    return Recommendation(rationale="", config=ConfigVector(
        executor_cores=4, executor_memory_gb=14, executor_instances=4, driver_memory_gb=8,
        shuffle_partitions=400, advisory_partition_mb=64, aqe_enabled=True,
        broadcast_threshold_mb=10))

def test_distill_and_leakage_filter(tmp_path):
    ms = MemoryStore(tmp_path / "mem.jsonl")
    out = Outcome(status="completed", per_query={"q1": QueryResult(status="success", duration_ms=1000)})
    ms.distill(_ep("sf100", "q1"), _rec(), out)
    ms.distill(_ep("sf250", "q2"), _rec(), out)
    hits = ms.retrieve(_ep("sf100", "q1"))
    assert len(hits) == 1 and "sf250" in hits[0]      # memoria do proprio (sf100,q1) excluida
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/memory/store.py
import json
from pathlib import Path
from advisor_loop.episode import Episode, Outcome, Recommendation

class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def distill(self, ep: Episode, rec: Recommendation, out: Outcome) -> dict:
        qstat = {q: r.status for q, r in out.per_query.items()}
        mem = {"workload": ep.workload, "dataset": ep.dataset, "queries": ep.queries,
               "config": rec.config.model_dump(), "status": out.status,
               "lesson": f"[{ep.dataset} {','.join(ep.queries)}] config "
                         f"{rec.config.executor_cores}c/{rec.config.executor_memory_gb}g"
                         f"x{rec.config.executor_instances} -> {out.status} ({qstat})"}
        with open(self.path, "a") as f:
            f.write(json.dumps(mem, ensure_ascii=False) + "\n")
        return mem

    def retrieve(self, ep: Episode, k: int = 5) -> list[str]:
        def leaks(m):     # anti-vazamento: nada do proprio (dataset, query)
            return m["dataset"] == ep.dataset and set(m["queries"]) & set(ep.queries)
        def score(m):
            return (m["workload"] == ep.workload, m["dataset"] == ep.dataset)
        pool = [m for m in self._all() if not leaks(m)]
        pool.sort(key=score, reverse=True)
        return [m["lesson"] for m in pool[:k]]
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: memory store with leave-one-out leakage filter"`

---

### Task 10: CampaignRunner + CLI (`runner.py`)

**Files:**
- Create: `advisor_loop/runner.py`, `campaigns/pilot-replay.yaml`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: TODOS os componentes anteriores (Tasks 1–9).
- Produces: `run_campaign(campaign_path: Path, deps: Deps) -> dict` (sumário: episódios rodados/pulados, custo) e `main()` (CLI: `advisor-loop run campaigns/x.yaml`). `Deps` = dataclass com `client, journal, pack, guardian, backend, memory, probe_factory` (injeção → testes com fakes). YAML de campanha: `name, level, provider, datasets: [..], queries: [..], objective, tries, memory: on|off`. Ciclo por episódio (spec §3.3): render → retrieve (se L2) → advise (probe se L1+) → guardian (1 retry com motivo) → execute → journal `query_result` por query (task_key!) → distill (se L2) → `episode_end`. Episódio cujo primeiro task_key já está no Journal é PULADO (retomada).

- [ ] **Step 1: Teste falhando**

```python
# tests/test_runner.py
import json, yaml
from advisor_loop.runner import run_campaign, Deps
from advisor_loop.journal import Journal
from advisor_loop.guardian import Guardian
from advisor_loop.memory.store import MemoryStore
from advisor_loop.advisor.provider import LLMResponse
from tests.test_guardian import CLUSTER
from tests.test_advisor import _submit_call, FakeClient
from tests.test_replay import _ref
from advisor_loop.backends.replay import ReplayBackend

class Pack:
    hash = "deadbeefdeadbeef"
    def render(self, ep):
        return f"ctx {ep.dataset}"

def _campaign(tmp_path):
    c = {"name": "pilot", "level": "L0", "provider": "m", "datasets": ["sf100"],
         "queries": ["q1"], "objective": "min_time", "tries": 1, "memory": "off"}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c))
    return p

def _deps(tmp_path, scripted):
    return Deps(client=FakeClient(scripted), journal=Journal(tmp_path / "j.jsonl"),
                pack=Pack(), guardian=Guardian(CLUSTER),
                backend=ReplayBackend(_ref(tmp_path)),
                memory=MemoryStore(tmp_path / "m.jsonl"), probe_factory=lambda ep: None)

def test_run_and_resume(tmp_path):
    camp = _campaign(tmp_path)
    s1 = run_campaign(camp, _deps(tmp_path, [_submit_call()]))
    assert s1["episodes_run"] == 1
    s2 = run_campaign(camp, _deps(tmp_path, []))       # sem chamadas: tudo pulado
    assert s2["episodes_skipped"] == 1 and s2["episodes_run"] == 0
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/runner.py
import argparse, sys
from dataclasses import dataclass
from pathlib import Path
import yaml
from advisor_loop.advisor.advisor import Advisor
from advisor_loop.episode import Episode, task_key

@dataclass
class Deps:
    client: object
    journal: object
    pack: object
    guardian: object
    backend: object
    memory: object
    probe_factory: object       # (episode) -> BudgetedProbe | None

def run_campaign(campaign_path: str | Path, deps: Deps) -> dict:
    camp = yaml.safe_load(Path(campaign_path).read_text())
    adv = Advisor(deps.client, deps.journal)
    ran = skipped = 0
    cost = 0.0
    for dataset in camp["datasets"]:
        for query in camp["queries"]:
            for try_n in range(1, camp.get("tries", 1) + 1):
                ep = Episode(workload=camp["name"], dataset=dataset, queries=[query],
                             objective=camp.get("objective", "min_time"),
                             level=camp["level"], provider=camp["provider"])
                probe = deps.probe_factory(ep) if camp["level"] in ("L1", "L2") else None
                mems = deps.memory.retrieve(ep) if camp.get("memory") == "on" else []
                context = deps.pack.render(ep)
                rec = adv.advise(ep, context, probe, mems)
                tk = task_key(ep.workload, dataset, query, rec.config.config_hash(), try_n)
                if deps.journal.has(tk):
                    skipped += 1
                    continue
                ok, reason = deps.guardian.validate(rec.config)
                if not ok:                              # 1 retry com feedback
                    rec = adv.advise(ep, context + f"\n\nREJEITADA: {reason}", probe, mems)
                    ok, reason = deps.guardian.validate(rec.config)
                if not ok:
                    deps.journal.append({"kind": "episode_end", "task_key": tk,
                                         "status": "invalid_recommendation", "reason": reason})
                    ran += 1
                    continue
                out = deps.backend.execute(ep, rec)
                for q, qr in out.per_query.items():
                    deps.journal.append({"kind": "query_result", "task_key": tk, "query": q,
                                         **qr.model_dump()})
                deps.journal.append({"kind": "episode_end", "task_key": tk,
                                     "status": out.status, "pack_hash": deps.pack.hash,
                                     "config": rec.config.model_dump()})
                if camp.get("memory") == "on":
                    deps.memory.distill(ep, rec, out)
                ran += 1
                cost += out.cost_usd
    return {"episodes_run": ran, "episodes_skipped": skipped, "cost_usd": cost}

def main() -> int:
    ap = argparse.ArgumentParser(prog="advisor-loop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("campaign")
    args = ap.parse_args()
    from advisor_loop.wiring import build_deps          # Task 13
    summary = run_campaign(args.campaign, build_deps(Path(args.campaign)))
    print(summary)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Nota: a chave de retomada usa o config_hash da recomendação — em L0/L1 com temperatura fixa a recomendação repetida gera a mesma chave; o custo de re-advise num resume é aceito e logado (limitação declarada; alternativa por episode-id documentada como melhoria futura).

Criar também `campaigns/pilot-replay.yaml`:

```yaml
name: pilot-replay
level: L0
provider: deepseek/deepseek-v4-flash-0731
datasets: [sf100]
queries: [q1, q5, q17]
objective: min_time
tries: 1
memory: "off"
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: campaign runner with resume + pilot campaign"`

---

### Task 11: Supervisor com filtro heurístico (`supervisor.py`)

**Files:**
- Create: `advisor_loop/supervisor.py`
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: `Decision` (Task 1), `OpenRouterClient.chat` (Task 5), `Journal` (Task 2).
- Produces: `SupervisorPolicy(decision_interval_s=600, min_observation_s=600, max_wall_clock_s=28800, gc_ratio_trigger=0.4, stall_s=600)`; `Supervisor(client, journal, policy, model="openai/gpt-oss-20b")` com `check(snapshot: dict, elapsed_s: float) -> Decision`. `snapshot` = `{stages_completed, stages_total, tasks_completed, gc_time_ms, run_time_ms, executors_lost, last_progress_s}`. Regras determinísticas ANTES do LLM: `elapsed < min_observation` → continue sem LLM; `elapsed > max_wall_clock` → kill sem LLM (backstop); nenhum gatilho (progresso recente, GC ok, sem executor perdido) → continue sem LLM; gatilho disparado → chama LLM com snapshot e devolve `Decision` (JSON via tool `submit_decision`).

- [ ] **Step 1: Teste falhando**

```python
# tests/test_supervisor.py
import json
from advisor_loop.supervisor import Supervisor, SupervisorPolicy
from advisor_loop.advisor.provider import LLMResponse
from advisor_loop.journal import Journal
from tests.test_advisor import FakeClient

SNAP_OK = {"stages_completed": 5, "stages_total": 10, "tasks_completed": 100,
           "gc_time_ms": 100, "run_time_ms": 10000, "executors_lost": 0, "last_progress_s": 5}
SNAP_STALL = {**SNAP_OK, "executors_lost": 2, "last_progress_s": 900}

def _kill_call():
    args = json.dumps({"action": "kill", "reason": "sem progresso",
                       "evidence": ["2 executors perdidos"], "confidence": 0.9})
    return LLMResponse(message={}, tool_calls=[
        {"id": "1", "function": {"name": "submit_decision", "arguments": args}}])

def test_heuristic_paths_no_llm(tmp_path):
    sup = Supervisor(FakeClient([]), Journal(tmp_path / "j.jsonl"), SupervisorPolicy())
    assert sup.check(SNAP_STALL, elapsed_s=60).action == "continue"     # janela minima
    assert sup.check(SNAP_OK, elapsed_s=5000).action == "continue"      # tudo normal
    assert sup.check(SNAP_OK, elapsed_s=99999).action == "kill"         # hard cap

def test_trigger_calls_llm(tmp_path):
    sup = Supervisor(FakeClient([_kill_call()]), Journal(tmp_path / "j.jsonl"), SupervisorPolicy())
    d = sup.check(SNAP_STALL, elapsed_s=5000)
    assert d.action == "kill" and d.confidence == 0.9
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/supervisor.py
import json
from dataclasses import dataclass
from advisor_loop.episode import Decision
from advisor_loop.journal import Journal

@dataclass
class SupervisorPolicy:
    decision_interval_s: int = 600
    min_observation_s: int = 600
    max_wall_clock_s: int = 28800
    gc_ratio_trigger: float = 0.4
    stall_s: int = 600

_DECIDE = {"type": "function", "function": {
    "name": "submit_decision",
    "parameters": {"type": "object",
                   "required": ["action", "reason", "evidence", "confidence"],
                   "properties": {"action": {"enum": ["continue", "kill"]},
                                  "reason": {"type": "string"},
                                  "evidence": {"type": "array", "items": {"type": "string"}},
                                  "confidence": {"type": "number"}}}}}

class Supervisor:
    def __init__(self, client, journal: Journal, policy: SupervisorPolicy,
                 model: str = "openai/gpt-oss-20b"):
        self.client, self.journal, self.policy, self.model = client, journal, policy, model

    def _triggered(self, s: dict) -> list[str]:
        t = []
        if s["executors_lost"] > 0:
            t.append(f"{s['executors_lost']} executors perdidos")
        if s["last_progress_s"] > self.policy.stall_s:
            t.append(f"sem progresso ha {s['last_progress_s']}s")
        if s["run_time_ms"] and s["gc_time_ms"] / s["run_time_ms"] > self.policy.gc_ratio_trigger:
            t.append("GC acima do limiar")
        return t

    def check(self, snapshot: dict, elapsed_s: float) -> Decision:
        if elapsed_s > self.policy.max_wall_clock_s:
            d = Decision(action="kill", reason="hard cap de wall-clock (backstop de orcamento)",
                         evidence=[f"elapsed={elapsed_s}s"], confidence=1.0)
        elif elapsed_s < self.policy.min_observation_s:
            d = Decision(action="continue", reason="janela minima de observacao",
                         evidence=[], confidence=1.0)
        else:
            triggers = self._triggered(snapshot)
            if not triggers:
                d = Decision(action="continue", reason="sem gatilhos", evidence=[], confidence=1.0)
            else:
                r = self.client.chat(self.model, [
                    {"role": "system", "content":
                     "Decida continuar ou matar um job de query distribuida. "
                     "Responda SEMPRE via submit_decision."},
                    {"role": "user", "content":
                     f"Gatilhos: {triggers}\nSnapshot: {json.dumps(snapshot)}\n"
                     f"Tempo decorrido: {elapsed_s}s"}], tools=[_DECIDE])
                args = json.loads(r.tool_calls[0]["function"]["arguments"])
                d = Decision(**args)
        self.journal.append({"kind": "decision", **d.model_dump(), "elapsed_s": elapsed_s})
        return d
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: supervisor with heuristic-first LLM decisions"`

---

### Task 12: Importador do dataset de referência (`scripts/import_reference.py`)

**Files:**
- Create: `scripts/import_reference.py`
- Test: `tests/test_import_reference.py`

**Interfaces:**
- Consumes: schema do parquet de referência (Task 8).
- Produces: CLI `python scripts/import_reference.py <dir_de_jsons> <saida.parquet>`. Formato de entrada documentado (benchmark JSON externo): objeto com `configuration{scale_factor, executor_cores, executor_memory, workers}` e `detailed_results[{query, try, status, execution_time_ms}]`. Converte `executor_memory` "14g" → 14; `scale_factor` 100 → dataset "sf100". Um registro por (query, try).

- [ ] **Step 1: Teste falhando**

```python
# tests/test_import_reference.py
import duckdb, json
from scripts.import_reference import convert_dir

def test_convert(tmp_path):
    src = {"configuration": {"scale_factor": 100, "executor_cores": "4",
                             "executor_memory": "14g", "workers": 4},
           "detailed_results": [{"query": "q1", "try": 1, "status": "success",
                                 "execution_time_ms": 1234.5}]}
    (tmp_path / "b.json").write_text(json.dumps(src))
    out = tmp_path / "ref.parquet"
    n = convert_dir(tmp_path, out)
    assert n == 1
    row = duckdb.sql(f"SELECT * FROM read_parquet('{out}')").fetchone()
    assert row == ("sf100", "q1", 4, 14, 4, "success", 1234.5)
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# scripts/import_reference.py
import json, re, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def convert_dir(src_dir: str | Path, out: str | Path) -> int:
    rows = []
    for f in sorted(Path(src_dir).rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        c = d.get("configuration") or {}
        if not c or "detailed_results" not in d:
            continue
        mem = int(re.sub(r"\D", "", str(c["executor_memory"])))
        for r in d["detailed_results"]:
            rows.append({"dataset": f"sf{c['scale_factor']}", "query": r["query"],
                         "executor_cores": int(c["executor_cores"]),
                         "executor_memory_gb": mem,
                         "executor_instances": int(c["workers"]),
                         "status": r["status"],
                         "duration_ms": r.get("execution_time_ms")})
    pq.write_table(pa.Table.from_pylist(rows), str(out))
    return len(rows)

if __name__ == "__main__":
    print(convert_dir(sys.argv[1], sys.argv[2]))
```

- [ ] **Step 4: Rodar e ver passar** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: reference dataset importer"`

---

### Task 13: Archiver + wiring de produção (`archiver.py`, `wiring.py`)

**Files:**
- Create: `advisor_loop/archiver.py`, `advisor_loop/wiring.py`, `.env.example`
- Test: `tests/test_archiver.py`

**Interfaces:**
- Consumes: `Journal` (Task 2); componentes de Tasks 3–10 (wiring).
- Produces: `Archiver(repo_root)` com `publish(campaign_name, journal_path, extra: dict) -> Path` — copia journal para `evidence/<campanha>/journal.jsonl.zst` (zstandard), escreve `summary.json` (extra), roda `scan_secrets(text) -> list[str]` (padrões `sk-or-`, `sk-ant-`, `AKIA`, `-----BEGIN`) em TODO conteúdo antes de `git add/commit` via subprocess; segredo encontrado → `SecretDetected`, nada é commitado. `build_deps(campaign_path) -> Deps` monta produção: OpenRouterClient real, Journal em `evidence/<campanha>/journal.jsonl`, KnowledgePack de `packs/`, Guardian do pack, ReplayBackend de `data/reference.parquet`, MemoryStore em `evidence/<campanha>/memory.jsonl`, probe_factory com DuckDBProbe de `datasets.yaml` (mapa nome→paths por dataset).

- [ ] **Step 1: Teste falhando**

```python
# tests/test_archiver.py
import subprocess, pytest
from advisor_loop.archiver import Archiver, SecretDetected, scan_secrets
from advisor_loop.journal import Journal

def test_scan_secrets():
    assert scan_secrets("nada aqui") == []
    assert scan_secrets("key=sk-or-v1-abc") != []

def test_publish_compresses_and_commits(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    j = Journal(tmp_path / "j.jsonl")
    j.append({"kind": "episode_end", "status": "completed"})
    out = Archiver(tmp_path).publish("pilot", j.path, {"episodes_run": 1}, push=False)
    assert (out / "journal.jsonl.zst").exists() and (out / "summary.json").exists()
    log = subprocess.run(["git", "-C", str(tmp_path), "log", "--oneline"],
                         capture_output=True, text=True).stdout
    assert "evidence: pilot" in log

def test_publish_blocks_secrets(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    j = Journal(tmp_path / "j.jsonl")
    j.append({"kind": "llm_call", "note": "sk-or-v1-LEAKED"})
    with pytest.raises(SecretDetected):
        Archiver(tmp_path).publish("pilot", j.path, {}, push=False)
```

- [ ] **Step 2: Rodar e ver falhar** → FAIL

- [ ] **Step 3: Implementar**

```python
# advisor_loop/archiver.py
import json, re, subprocess
from pathlib import Path
import zstandard

SECRET_PATTERNS = [r"sk-or-[\w-]+", r"sk-ant-[\w-]+", r"AKIA[0-9A-Z]{16}", r"-----BEGIN"]

class SecretDetected(Exception):
    pass

def scan_secrets(text: str) -> list[str]:
    return [p for p in SECRET_PATTERNS if re.search(p, text)]

class Archiver:
    def __init__(self, repo_root: str | Path):
        self.root = Path(repo_root)

    def publish(self, campaign: str, journal_path: Path, extra: dict, push: bool = True) -> Path:
        raw = Path(journal_path).read_text()
        summary = json.dumps(extra, ensure_ascii=False, indent=1)
        if scan_secrets(raw) or scan_secrets(summary):
            raise SecretDetected("segredo detectado; publicacao abortada")
        out = self.root / "evidence" / campaign
        out.mkdir(parents=True, exist_ok=True)
        (out / "journal.jsonl.zst").write_bytes(zstandard.compress(raw.encode()))
        (out / "summary.json").write_text(summary)
        rel = out.relative_to(self.root)
        subprocess.run(["git", "-C", str(self.root), "add", str(rel)], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-m",
                        f"evidence: {campaign}"], check=True)
        if push:
            subprocess.run(["git", "-C", str(self.root), "push"], check=True)
        return out
```

```python
# advisor_loop/wiring.py — montagem de producao (sem teste unitario; validada no smoke do Task 14)
from pathlib import Path
import yaml
from advisor_loop.advisor.provider import OpenRouterClient
from advisor_loop.backends.replay import ReplayBackend
from advisor_loop.guardian import Guardian
from advisor_loop.journal import Journal
from advisor_loop.knowledge.pack import KnowledgePack
from advisor_loop.memory.store import MemoryStore
from advisor_loop.probes.budget import BudgetedProbe
from advisor_loop.probes.duckdb_probe import DuckDBProbe
from advisor_loop.runner import Deps

def build_deps(campaign_path: Path, root: Path = Path(".")) -> Deps:
    camp = yaml.safe_load(campaign_path.read_text())
    ev = root / "evidence" / camp["name"]
    pack = KnowledgePack.load(root / "packs")
    tables_by_dataset = {}
    ds_file = root / "datasets.yaml"
    if ds_file.exists():
        tables_by_dataset = yaml.safe_load(ds_file.read_text()) or {}

    def probe_factory(ep):
        tables = tables_by_dataset.get(ep.dataset)
        return BudgetedProbe(DuckDBProbe(tables)) if tables else None

    return Deps(client=OpenRouterClient(), journal=Journal(ev / "journal.jsonl"),
                pack=pack, guardian=Guardian(pack.cluster),
                backend=ReplayBackend(root / "data" / "reference.parquet"),
                memory=MemoryStore(ev / "memory.jsonl"), probe_factory=probe_factory)
```

```bash
# .env.example
OPENROUTER_API_KEY=sk-or-v1-SUA-CHAVE-AQUI
```

- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_archiver.py -v` → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: archiver with secret scan + production wiring"`

---

### Task 14: Smoke de ponta a ponta + PR

**Files:**
- Modify: `README.md` (instruções de uso)

**Interfaces:**
- Consumes: tudo.
- Produces: piloto real executado (1 episódio L0 via OpenRouter no DeepSeek V4 Flash, com replay), evidência publicada, PR aberto.

- [ ] **Step 1: Suite completa verde** — `pytest -v` → todos PASS

- [ ] **Step 2: Gerar referência mínima e rodar o smoke real** (usa ~US$0,01 do crédito)

```bash
python scripts/import_reference.py <dir-com-jsons-de-referencia> data/reference.parquet
cp .env.example .env  # e colar a chave real (arquivo ja esta no .gitignore)
advisor-loop run campaigns/pilot-replay.yaml
# esperado: {'episodes_run': 3, 'episodes_skipped': 0, 'cost_usd': ...}
advisor-loop run campaigns/pilot-replay.yaml
# esperado: episodes_skipped == 3 (retomada idempotente)
```

- [ ] **Step 3: Publicar evidência** — verificar `evidence/pilot-replay/` commitado (journal.jsonl.zst + summary.json), sem segredos.

- [ ] **Step 4: README com quickstart** (instalação, .env, import, run, layout do repo — 30 linhas).

- [ ] **Step 5: Commit final + PR**

```bash
git add -A && git commit -m "docs: quickstart README"
git push -u origin feat/core
gh pr create --draft --title "advisor-loop core: episode harness (replay-first)" \
  --body "Implements docs/spec.md Tasks 0-14: journal, probes, advisor, guardian, replay, memory, runner, supervisor, archiver."
```

---

## Self-Review (executada)

1. **Cobertura do spec**: componentes §3.1 → Tasks 1–13 (KnowledgePack=3, Probe/Budget=4, Advisor=5–6, Guardian=7, ReplayBackend=8, Journal=2, MemoryStore=9, CampaignRunner=10, Supervisor=11, Archiver=13). Ciclo §3.3 → Task 10. Supervisor §3.4 (heurística-first, janela mínima, hard cap) → Task 11. Anti-vazamento §4 → Task 9 (filtro leave-one-out). Política de evidência §5 → Tasks 2+13. Resiliência §6: journal-first+idempotência (2, 10), secret scan (13). **Fora deste plano (plano 2, pós-piloto)**: `SparkK8sBackend` vivo, preflight de manifest do storage, lock de cluster, coleta real de snapshot via REST — dependem de acesso a cluster e entram com o go/no-go de F3 do spec; o Supervisor já nasce pronto para recebê-los.
2. **Placeholders**: nenhum "TBD/TODO"; todo step tem código ou comando concreto.
3. **Consistência de tipos**: `ConfigVector`/`Outcome`/`Decision` definidos no Task 1 e usados com os mesmos nomes em 6–11; `Deps` do Task 10 é o contrato do wiring no Task 13; schema do parquet do Task 8 é o contrato do importador no Task 12.
