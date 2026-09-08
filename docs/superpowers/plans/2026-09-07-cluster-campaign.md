# Cluster Campaign (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the closed loop on the real RNP cluster (TPC-DS SF100, q03/q23/q50) as a falsifiable Phase-2 study: free-schema LLM vs free Random, both under B=8/k=3, measured against the dissertation's matrix oracle.

**Architecture:** Reuse `loop.optimize` + `SparkK8sBackend` + `runner._run_optimize_campaign`. Add: k-tries/median to the loop, a free-schema random advisor, a free-range knowledge pack, arm selection in the optimize campaign, and offline oracle/report analysis. Cluster config via YAML.

**Tech Stack:** Python 3.11+, pydantic, pyyaml, duckdb (analysis), pytest. No new runtime deps.

**Spec:** `docs/superpowers/specs/2026-09-07-cluster-campaign-design.md`

## Global Constraints

- Advisor interface is fixed: `advise(ep, context, probe, memories, history=[]) -> Recommendation`. New advisors MUST conform.
- `ConfigVector` field ranges are the schema of record (episode.py). Free proposals sample/validate within these.
- L0 only: `probe=None` in the loop for this campaign. Do not enable probe tools.
- LLM arm: `deepseek/deepseek-v4-flash-0731`, reasoning disabled, `tool_choice="required"`, `max_tokens=1024`.
- Matrix data is read-only; never launch a cluster benchmark from the analysis path. Clean source only (rerun-202608/202609 / consolidated parquet), never `gold/legacy-pre-repair`.
- One cluster run at a time (contamination rule). Runner is serial.
- All new code has unit tests that run without cluster or API access.

---

### Task 1: k-tries + baseline in `loop.optimize`

**Files:**
- Modify: `advisor_loop/loop.py`
- Test: `tests/test_loop_tries.py`

**Interfaces:**
- Consumes: `deps.backend.execute(ep, rec) -> Outcome`, `Outcome.per_query[q].duration_ms`.
- Produces: `optimize(episode, deps, advisor, max_iters=4, patience=2, tries=1) -> dict` with added keys `baseline_ms` and per-iteration `durations_ms` (list) + `duration_ms` (median). Default config measured as iteration 0.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_loop_tries.py
import statistics
from advisor_loop.loop import optimize
from advisor_loop.episode import ConfigVector, Episode, Outcome, QueryResult, Recommendation

DEFAULT = dict(executor_cores=8, executor_memory_gb=8, executor_instances=1,
               driver_memory_gb=4, shuffle_partitions=200, advisory_partition_mb=64,
               aqe_enabled=True, broadcast_threshold_mb=10)

def _outcome(ms):
    return Outcome(status="completed", per_query={"q": QueryResult(status="success", duration_ms=ms)})

class _ScriptedBackend:
    """Returns durations from a per-config-hash queue; records call order."""
    def __init__(self, script):  # script: list of durations, consumed in order
        self.script = list(script); self.calls = []
    def execute(self, ep, rec):
        ms = self.script.pop(0); self.calls.append((rec.config.config_hash(), ms))
        return _outcome(ms)

class _FixedAdvisor:
    def __init__(self, cfg): self.cfg = cfg
    def advise(self, ep, context, probe, mems, history=[]):
        return Recommendation(config=self.cfg, rationale="x")

class _Deps:
    class _J:
        def append(self, *a, **k): pass
    class _M:
        def retrieve(self, ep): return []
        def distill(self, *a, **k): pass
    class _P:
        def render(self, ep): return "ctx"
    class _G:
        def validate(self, cfg): return True, "ok"
    def __init__(self, backend):
        self.backend, self.journal, self.memory = backend, self._J(), self._M()
        self.pack, self.guardian = self._P(), self._G()
        self.probe_factory = lambda ep: None

def _ep():
    return Episode(workload="w", dataset="sf100", queries=["q"], provider="m")

def test_baseline_is_default_median():
    # iter0 default k=3 → median(30,10,20)=20 ; iter1 proposal k=3 → median(9,7,8)=8
    be = _ScriptedBackend([30, 10, 20, 9, 7, 8])
    cfg = ConfigVector(**{**DEFAULT, "shuffle_partitions": 400})
    res = optimize(_ep(), _Deps(be), _FixedAdvisor(cfg), max_iters=1, patience=1, tries=3)
    assert res["baseline_ms"] == 20
    assert res["best_duration_ms"] == 8
    assert res["iterations"][0]["duration_ms"] == 8
    assert res["iterations"][0]["durations_ms"] == [9, 7, 8]

def test_median_drives_best_tracking():
    # two proposals; medians 8 then 12 → best stays 8
    be = _ScriptedBackend([100, 100, 100,  9, 7, 8,  13, 12, 11])
    class _TwoAdvisor:
        def __init__(self): self.n = 0
        def advise(self, ep, context, probe, mems, history=[]):
            self.n += 1
            sp = 300 if self.n == 1 else 500
            return Recommendation(config=ConfigVector(**{**DEFAULT, "shuffle_partitions": sp}), rationale="x")
    res = optimize(_ep(), _Deps(be), _TwoAdvisor(), max_iters=2, patience=2, tries=3)
    assert res["best_duration_ms"] == 8
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_loop_tries.py -v` → FAIL (`optimize` has no `tries`, no `baseline_ms`).

- [ ] **Step 3: Implement**

In `loop.py`, add a helper that runs the backend `tries` times and returns
`(median_ms, durations, representative_metrics)`; the representative metrics come
from the try whose duration is the median. Measure `DEFAULT_CFG` as iteration 0
into `baseline_ms`. Keep the existing propose→validate→execute→distill flow, but
replace the single `execute` with the median helper, and record `durations_ms` +
`samples` in each iteration dict. `DEFAULT_CFG` constant lives in `loop.py`:

```python
DEFAULT_CFG = ConfigVector(executor_cores=8, executor_memory_gb=8, executor_instances=1,
                           driver_memory_gb=4, shuffle_partitions=200, advisory_partition_mb=64,
                           aqe_enabled=True, broadcast_threshold_mb=10)

def _run_median(deps, ep, rec, tries):
    durs, metrics_by = [], {}
    for _ in range(tries):
        out = deps.backend.execute(ep, rec)
        for qr in out.per_query.values():
            if qr.status == "success" and qr.duration_ms is not None:
                durs.append(qr.duration_ms); metrics_by[qr.duration_ms] = qr.metrics or {}
                break
    if not durs:
        return None, [], {}
    med = statistics.median(durs)
    # representative metrics: closest recorded duration to the median
    key = min(metrics_by, key=lambda d: abs(d - med))
    return med, durs, metrics_by[key]
```

Baseline: before the loop, `bmed, bdurs, _ = _run_median(deps, ep, Recommendation(config=DEFAULT_CFG, rationale="baseline"), tries)`; store as `baseline_ms`. Inside the loop use `_run_median` for each validated proposal; feed `med` as `dur` to the existing best-tracking/patience logic. Add `durations_ms`/`samples` to each `iterations[]` entry. Return dict gains `"baseline_ms": bmed`.

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_loop_tries.py -v` → PASS.

- [ ] **Step 5: Run full suite** — `pytest -q` (ensure `tries=1` default keeps existing loop tests green).

- [ ] **Step 6: Commit** — `git add advisor_loop/loop.py tests/test_loop_tries.py && git commit -m "feat(loop): k-tries median + default baseline in optimize"`

---

### Task 2: `FreeRandomAdvisor`

**Files:**
- Modify: `advisor_loop/studies/advisors.py`
- Test: `tests/test_free_random_advisor.py`

**Interfaces:**
- Consumes: `ConfigVector` field ranges; `Guardian.validate(cfg) -> (bool, str)`.
- Produces: `FreeRandomAdvisor(guardian, seed=0)` with `advise(ep, context, probe, memories, history=[]) -> Recommendation`. Proposals uniform in ranges, feasible under `guardian`, deterministic per `(seed, dataset, query, len(history))`, no repeat within an episode.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_free_random_advisor.py
from advisor_loop.studies.advisors import FreeRandomAdvisor
from advisor_loop.episode import ConfigVector, Episode
from advisor_loop.guardian import Guardian

CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}

def _ep(): return Episode(workload="w", dataset="sf100", queries=["q23"], provider="m")

def _ranges_ok(c: ConfigVector):
    return (1 <= c.executor_cores <= 24 and 4 <= c.executor_memory_gb <= 96
            and 1 <= c.executor_instances <= 20 and 4 <= c.driver_memory_gb <= 16
            and 100 <= c.shuffle_partitions <= 4000 and 16 <= c.advisory_partition_mb <= 256
            and -1 <= c.broadcast_threshold_mb <= 512)

def test_all_proposals_feasible_and_in_range():
    g = Guardian(CAPS); adv = FreeRandomAdvisor(g, seed=1)
    hist = []
    for _ in range(8):
        rec = adv.advise(_ep(), "ctx", None, [], history=hist)
        assert _ranges_ok(rec.config)
        assert g.validate(rec.config)[0]
        hist.append((rec.config, None))

def test_deterministic():
    g = Guardian(CAPS)
    a, b = FreeRandomAdvisor(g, seed=7), FreeRandomAdvisor(g, seed=7)
    r1 = a.advise(_ep(), "c", None, [], history=[]).config
    r2 = b.advise(_ep(), "c", None, [], history=[]).config
    assert r1.config_hash() == r2.config_hash()

def test_no_repeat_within_episode():
    g = Guardian(CAPS); adv = FreeRandomAdvisor(g, seed=3)
    seen, hist = set(), []
    for _ in range(8):
        rec = adv.advise(_ep(), "c", None, [], history=hist)
        h = rec.config.config_hash()
        assert h not in seen; seen.add(h); hist.append((rec.config, None))
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_free_random_advisor.py -v` → FAIL (ImportError).

- [ ] **Step 3: Implement** in `advisors.py`:

```python
class FreeRandomAdvisor:
    """Free-schema random search: samples each ConfigVector field uniformly in
    its range, rejection-sampling against a Guardian so every proposal is
    feasible. Deterministic per (seed, dataset, query, len(history)); avoids
    repeats within an episode. Same search space as the LLM arm -> fair control."""
    _INT = {"executor_cores": (1, 24), "executor_memory_gb": (4, 96),
            "executor_instances": (1, 20), "driver_memory_gb": (4, 16),
            "shuffle_partitions": (100, 4000), "advisory_partition_mb": (16, 256),
            "broadcast_threshold_mb": (-1, 512)}

    def __init__(self, guardian, seed: int = 0):
        self.guardian, self.seed = guardian, seed

    def _sample(self, rng) -> ConfigVector:
        f = {k: rng.randint(lo, hi) for k, (lo, hi) in self._INT.items()}
        f["aqe_enabled"] = rng.random() < 0.5
        return ConfigVector(**f)

    def advise(self, ep, context, probe, memories, history=[]) -> Recommendation:
        seen = {c.config_hash() for c, _ in history}
        rng = random.Random(f"{self.seed}|{ep.dataset}|{ep.queries[0]}|{len(history)}")
        for _ in range(2000):
            cfg = self._sample(rng)
            if self.guardian.validate(cfg)[0] and cfg.config_hash() not in seen:
                return Recommendation(config=cfg, rationale=f"free-random trial #{len(history)+1}")
        raise RuntimeError("FreeRandomAdvisor: no feasible unique config in 2000 tries")
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_free_random_advisor.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add advisor_loop/studies/advisors.py tests/test_free_random_advisor.py && git commit -m "feat(advisors): free-schema random search control"`

---

### Task 3: Free-range knowledge pack

**Files:**
- Create: `packs/cluster-rnp.yaml`
- Modify: `packs/resources.yaml` (neutralize profile-menu wording), `packs/executor-profiles.yaml` (remove or replace with range framing)
- Test: `tests/test_pack_free_range.py`

**Interfaces:**
- Consumes: `KnowledgePack.load(packs).render(episode)`, `.cluster`.
- Produces: rendered context contains knob **ranges** + caps and the free-proposal instruction; contains no "choose one of these profiles" text. `.cluster` returns the RNP caps.

- [ ] **Step 1: Write failing test**

```python
# tests/test_pack_free_range.py
from pathlib import Path
from advisor_loop.knowledge.pack import KnowledgePack
from advisor_loop.episode import Episode

def test_free_range_pack(tmp_path=Path("packs")):
    pack = KnowledgePack.load("packs")
    txt = pack.render(Episode(workload="w", dataset="sf100", queries=["q23"], provider="m"))
    assert "executor_cores" in txt and "1" in txt and "24" in txt   # range surfaced
    low = txt.lower()
    assert "escolha um par" not in low and "destes perfis" not in low  # no menu constraint
    c = pack.cluster
    assert c["max_pods"] >= 21 and c["total_cores"] >= 120
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_pack_free_range.py -v` → FAIL (menu wording present / caps mismatch).

- [ ] **Step 3: Implement.** Create `packs/cluster-rnp.yaml` with the `cluster:` caps from spec §3.2 (max_pods 40, total_cores 120, total_memory_gb 400, max_cores_per_container 24, max_memory_gb_per_container 96). Rewrite `resources.yaml`/`executor-profiles.yaml` to present ranges + free-proposal instruction:

```yaml
# packs/knob-ranges.yaml (replaces executor-profiles.yaml menu framing)
proposal:
  freedom: >
    Voce escolhe LIVREMENTE qualquer valor dentro das faixas abaixo, respeitando
    os limites fisicos do cluster. Nao ha perfis pre-definidos a seguir; qualquer
    combinacao valida e permitida.
  knob_ranges:
    executor_cores: {min: 1, max: 24}
    executor_memory_gb: {min: 4, max: 96}
    executor_instances: {min: 1, max: 20}
    driver_memory_gb: {min: 4, max: 16}
    shuffle_partitions: {min: 100, max: 4000}
    advisory_partition_mb: {min: 16, max: 256}
    aqe_enabled: {values: [true, false]}
    broadcast_threshold_mb: {min: -1, max: 512, note: "-1 desliga broadcast"}
```

Remove `executor-profiles.yaml` (or replace its body with the above). Ensure exactly one YAML defines `cluster:` (cluster-rnp.yaml) — remove `cluster:` from cluster-example.yaml or delete that file to avoid `KnowledgePack.cluster` ambiguity. Keep `spark-3.5.yaml` knob semantics (T1) and `resources.yaml` cluster_resources description, but strip the "escolha um par (cores, memory_gb) destes perfis" lines.

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_pack_free_range.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add packs tests/test_pack_free_range.py && git commit -m "feat(pack): free-range knob framing + RNP cluster caps"`

---

### Task 4: Arm selection in the optimize campaign

**Files:**
- Modify: `advisor_loop/runner.py` (`_run_optimize_campaign`), `advisor_loop/wiring.py` (pass guardian to a random advisor factory)
- Test: `tests/test_optimize_campaign_arms.py`

**Interfaces:**
- Consumes: `camp["arm"]` (`"llm"` default | `"random"`), `camp["tries"]` (default 1), `deps.guardian`.
- Produces: `_run_optimize_campaign` builds `Advisor` for `llm` or `FreeRandomAdvisor(deps.guardian, seed)` for `random`, and passes `tries` into `optimize`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_optimize_campaign_arms.py
from pathlib import Path
import yaml
from advisor_loop.runner import _run_optimize_campaign, Deps

# minimal fakes: backend returns fixed durations, journal/memory/pack/guardian stubs
class _BE:
    def execute(self, ep, rec):
        from advisor_loop.episode import Outcome, QueryResult
        return Outcome(status="completed", per_query={"q23": QueryResult(status="success", duration_ms=100.0)})
class _J:
    def __init__(self): self.rows=[]
    def append(self, r): self.rows.append(r)
    def has(self, k): return False
class _M:
    def retrieve(self, ep): return []
    def distill(self, *a, **k): pass
class _P:
    def render(self, ep): return "ctx"

def test_random_arm_drives_loop(tmp_path):
    from advisor_loop.guardian import Guardian
    caps = {"max_cores_per_container":24,"max_memory_gb_per_container":96,
            "max_pods":40,"total_cores":120,"total_memory_gb":400}
    j=_J()
    deps=Deps(client=None, journal=j, pack=_P(), guardian=Guardian(caps),
              backend=_BE(), memory=_M(), probe_factory=lambda ep: None)
    camp={"name":"t","mode":"optimize","arm":"random","datasets":["sf100"],
          "queries":["q23"],"level":"L0","provider":"m","max_iters":3,"patience":2,"tries":2}
    out=_run_optimize_campaign(camp, deps, None)  # adv arg unused for random
    assert out["episodes_run"]==1
    assert any(r.get("kind")=="opt_result" for r in j.rows)
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_optimize_campaign_arms.py -v` → FAIL (no arm handling / tries).

- [ ] **Step 3: Implement.** In `_run_optimize_campaign`, select the advisor per `camp.get("arm","llm")`: `llm` → the passed `adv` (LLM `Advisor`); `random` → `FreeRandomAdvisor(deps.guardian, seed=camp.get("seed",0))`. Pass `tries=camp.get("tries",1)` to `optimize(...)`. In `run_campaign`, when `mode=="optimize"`, still build the LLM `adv` (used only if arm=llm). Record `arm` + `baseline_ms` in the `opt_result` journal row.

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_optimize_campaign_arms.py -v` → PASS; then `pytest -q`.

- [ ] **Step 5: Commit** — `git add advisor_loop/runner.py advisor_loop/wiring.py tests/test_optimize_campaign_arms.py && git commit -m "feat(runner): arm select (llm|random) + tries in optimize campaign"`

---

### Task 5: `analysis/oracle.py`

**Files:**
- Create: `analysis/__init__.py`, `analysis/oracle.py`
- Test: `tests/test_oracle.py`

**Interfaces:**
- Produces: `oracle_table(parquet_path) -> dict[(sf:int, query:str), float]` = min success `duration_ms`; `best_config(parquet_path, sf, query) -> dict` (winning row's cores/mem/workers/duration).

- [ ] **Step 1: Write failing test**

```python
# tests/test_oracle.py
import duckdb, pandas as pd
from analysis.oracle import oracle_table

def test_oracle_min_success(tmp_path):
    df = pd.DataFrame([
        {"scale_factor":100,"query":"q23","duration_ms":300.0,"status":"success"},
        {"scale_factor":100,"query":"q23","duration_ms":120.0,"status":"success"},
        {"scale_factor":100,"query":"q23","duration_ms":90.0,"status":"failed"},  # ignored
        {"scale_factor":100,"query":"q50","duration_ms":500.0,"status":"success"},
    ])
    p = tmp_path/"m.parquet"; df.to_parquet(p)
    t = oracle_table(str(p))
    assert t[(100,"q23")] == 120.0
    assert t[(100,"q50")] == 500.0
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_oracle.py -v` → FAIL.

- [ ] **Step 3: Implement** `oracle.py` using duckdb `read_parquet`, `min(duration_ms)` where `status='success'` group by `scale_factor, query`; `best_config` selects the argmin row.

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_oracle.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add analysis/oracle.py analysis/__init__.py tests/test_oracle.py && git commit -m "feat(analysis): matrix oracle extraction"`

---

### Task 6: `analysis/report.py`

**Files:**
- Create: `analysis/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: journal `opt_result` rows (`best_duration_ms`, `baseline_ms`, `dataset`, `query`, `arm`, `iterations`), oracle table.
- Produces: `regret(best_ms, t_star) -> float` (= best/t*−1, may be negative); `speedup(baseline_ms, best_ms) -> float`; `anytime(iterations) -> list[float]`; `build_report(opt_results, oracle) -> dict` (per (query,arm) rows + summary).

- [ ] **Step 1: Write failing test**

```python
# tests/test_report.py
from analysis.report import regret, speedup, anytime, build_report

def test_math():
    assert abs(regret(120, 100) - 0.2) < 1e-9
    assert regret(80, 100) < 0                 # beats oracle → negative
    assert abs(speedup(200, 100) - 2.0) < 1e-9

def test_anytime_monotone():
    iters=[{"duration_ms":300},{"duration_ms":120},{"duration_ms":150},{"duration_ms":90}]
    assert anytime(iters)==[300,120,120,90]

def test_build_report():
    ors={(100,"q23"):100.0}
    res=[{"dataset":"sf100","query":"q23","arm":"llm","best_duration_ms":120.0,
          "baseline_ms":240.0,"iterations":[{"duration_ms":120.0}]}]
    rep=build_report(res, ors)
    row=rep["rows"][0]
    assert abs(row["regret"]-0.2)<1e-9 and abs(row["speedup"]-2.0)<1e-9
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_report.py -v` → FAIL.

- [ ] **Step 3: Implement** the four functions; `build_report` maps `dataset` like `"sf100"` → int 100 to key the oracle, emits rows with regret/speedup/anytime + a summary (median regret per arm). Add a `__main__` that reads a journal jsonl + oracle parquet and prints/writes the table (covered by live use, not unit tested).

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_report.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add analysis/report.py tests/test_report.py && git commit -m "feat(analysis): regret/speedup/anytime report"`

---

### Task 7: Cluster config + matrix data + campaign YAMLs

**Files:**
- Create: `data/matrix/README.md`; copy `data/matrix/training_data.parquet`
- Create: `campaigns/cluster-smoke-sf100.yaml`, `campaigns/cluster-sf100.yaml`
- Modify: `datasets.yaml` (sf100 tables + s3 + spark ssh/image/node_selector)
- Test: `tests/test_campaign_configs.py`

**Interfaces:**
- Consumes: `wiring.build_deps` reads `datasets.yaml` (`s3`, `spark`) and campaign YAML (`backend: spark_k8s`, `mode: optimize`, `arm`, `tries`, `max_iters`, `patience`).

- [ ] **Step 1: Write failing test**

```python
# tests/test_campaign_configs.py
import yaml
from pathlib import Path

def test_smoke_campaign():
    c=yaml.safe_load(Path("campaigns/cluster-smoke-sf100.yaml").read_text())
    assert c["mode"]=="optimize" and c["backend"]=="spark_k8s"
    assert c["arm"]=="llm" and c["tries"]==1 and c["level"]=="L0"
    assert c["datasets"]==["sf100"] and set(c["queries"])=={"q03","q23","q50"}

def test_campaign():
    c=yaml.safe_load(Path("campaigns/cluster-sf100.yaml").read_text())
    assert c["tries"]==3 and c["max_iters"]==8 and c["arm"] in ("llm","random")
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_campaign_configs.py -v` → FAIL.

- [ ] **Step 3: Implement.** Copy the consolidated parquet:
`cp /home/italo/k8s-bigdata-bench/.claude/worktrees/banca-figuras/data/consolidated/training_data.parquet data/matrix/training_data.parquet` and write `data/matrix/README.md` (provenance: commit 01f7078, 68 cells / 20,196 execs, CLEAN dirs). Write the two campaign YAMLs:

```yaml
# campaigns/cluster-smoke-sf100.yaml
name: cluster-smoke-sf100
mode: optimize
backend: spark_k8s
arm: llm
level: L0
provider: deepseek/deepseek-v4-flash-0731
datasets: [sf100]
queries: [q03, q23, q50]
tries: 1
max_iters: 8
patience: 2
```

```yaml
# campaigns/cluster-sf100.yaml  (run once per arm; set arm: llm then arm: random)
name: cluster-sf100
mode: optimize
backend: spark_k8s
arm: llm            # flip to random for the control run
level: L0
provider: deepseek/deepseek-v4-flash-0731
datasets: [sf100]
queries: [q03, q23, q50]
tries: 3
max_iters: 8
patience: 2
```

Extend `datasets.yaml` with the `sf100` table map (TPC-DS table names → `s3a://silver/sf100/delta/parquet/<t>`), the `s3` MinIO block (endpoint `http://minio.italo-unb.svc.cluster.local:9000`, `minioadmin/minioadmin`, path-style), and a `spark` block: `ssh_cmd` (ssh … -p 2200 -i ~/.ssh/rnp), `image` (GHCR spark-advisor), optional `node_selector`. Do NOT put probe_service (L0). Do NOT commit `.env`.

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_campaign_configs.py -v` → PASS; then `pytest -q` (whole suite green).

- [ ] **Step 5: Commit** — `git add data/matrix campaigns datasets.yaml tests/test_campaign_configs.py && git commit -m "feat(config): RNP cluster campaign + smoke YAMLs + matrix oracle data"`

---

## Self-Review

- **Spec coverage:** §2.1 workloads → T7; §2.2 arms → T2,T4; §2.3 smoke → T7; §2.4 oracle → T5; §2.5 metrics → T6; §2.6 free-schema → T2,T3; §3.1 k-tries → T1; caps → T3. All covered.
- **Types:** `optimize(...tries=)`, `FreeRandomAdvisor(guardian, seed)`, `oracle_table`, `regret/speedup/anytime/build_report`, campaign keys `arm`/`tries` — consistent across tasks.
- **Placeholders:** none; every code step has concrete content.
- **Live run** (smoke, then campaign) is a manual step after T7, gated on user go — not a plan task (no cluster in CI).
