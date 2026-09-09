# Cluster Campaign (Phase 2, blind online tuning) — Design

**Status:** approved in brainstorming 2026-09-07
**Author:** Ítalo (with Claude)
**Repo:** `advisor-loop`, branch off `main`
**Related:** `docs/research-direction.md` (Phase 1 vs Phase 2), `docs/spec-closed-loop.md`
(SparkK8sBackend + OptimizationLoop), `experiments/phase2-pilot/` (SF1 local spike)

---

## 1. Goal

Run the closed optimization loop on the **real RNP cluster** with **real
workloads** (TPC-DS Silver in MinIO), and produce the first **falsifiable**
Phase-2 result: does a blind, online LLM advisor — proposing configs *freely*
within the schema — **match, approach, or beat the best configuration found by
the dissertation's exhaustive experimental matrix (exp05)**, in only a handful
of real runs?

This is the Phase-2 pivot made real: no discrete menu, no oracle handed to the
advisor, real Spark execution, and a non-LLM control under the same run budget.

**Non-goal (deferred):** DuckDB data probe (L1) — a later ablation ("does
data-aware probing improve recommendations?"). Bayesian optimization baseline —
a later increment ("is the LLM competitive with a SOTA tuner?"). Merging the
repaired-matrix branches into `main` of `k8s-bigdata-bench` (the user's call).

---

## 2. Scope

### 2.1 Workloads

- **Scale factor:** SF100 (TPC-DS Silver, `s3a://silver/sf100/delta/parquet`).
- **Queries:** `q03`, `q23`, `q50`.
  - Chosen for config sensitivity at SF100 (oracle spread 2.6×–3.6× between best
    and worst swept config → real tuning signal, unlike SF1) and for feasibility
    (they complete reliably; no q95-style artifacts).
  - The queries' "profiles" (q03 = scan+agg, q23/q50 = heavy joins) are for our
    understanding only. **The advisor never sees these labels** — it decides from
    the query's plan/telemetry/history + cluster caps.

### 2.2 Arms (same budget B, same k)

| Arm | Advisor | Search space | Purpose |
|-----|---------|--------------|---------|
| **LLM** | `Advisor` (L0) | free schema ∩ caps | the thing under test |
| **Random** | `FreeRandomAdvisor` | free schema ∩ caps | non-LLM control |

- **Budget B = 8** proposals per (query, arm).
- **k = 3** tries per proposal; feedback + best-tracking use the **median**
  duration (denoises the run-to-run variance that swamped the SF1 pilot).
- **BO arm: deferred** (separate future increment; not built here).

### 2.3 Smoke first

Before the full campaign, a **smoke**: LLM arm only, **k = 1**, the 3 queries.
Validates the real-cluster plumbing at SF100 quickly (~1–2 h), then the campaign
turns on `k = 3` and the Random arm.

### 2.4 Baseline: the experimental matrix (offline, zero cluster cost)

The dissertation's repaired matrix is the reference — **not** random/BO.

- **Source of truth:** `data/consolidated/training_data.parquet` (68 cells,
  20,196 executions) from `k8s-bigdata-bench` branch `worktree-banca-figuras`
  (identical on `worktree-exp05-rerun-matrix`). Raw clean campaigns:
  `experiments/05-.../metrics/rerun-202608/` + `rerun-202609/` (the dirs listed
  in `recommendation/config.py::CLEAN_METRICS_DIRS`). **Never** the
  `gold/`/`legacy-pre-repair/` data (pre-repair, stale).
- The consolidated parquet is copied once into this repo as
  `data/matrix/training_data.parquet` (analysis-only input; documented
  provenance) so the analysis is self-contained and reproducible offline.
- **Oracle** `t*(sf, query)` = `min(duration_ms)` over `status='success'` rows
  for that (sf, query). At SF100 for the targets (verified):
  - q03: t* ≈ 20.7 s (2c/7g × 20w) · q23: t* ≈ 92.8 s (12c/48g × 6w) ·
    q50: t* ≈ 53.1 s (12c/48g × 6w).

### 2.5 Metrics (no live oracle; computed post-hoc)

- **Regret vs matrix oracle:** `regret = t_arm_best / t* − 1`. Reconnects
  Phase-2 to Phase-1's regret metric, now live/blind. **Can be negative** — the
  free-schema advisor beating the exhaustive grid is a legitimate headline, not
  an error (see §2.6).
- **Speedup over default:** `t_default / t_arm_best`, where `t_default` is the
  Spark-default `ConfigVector` measured in-loop as iteration 0 (k tries, median).
- **Anytime-best curve:** best-so-far duration vs iteration index, per arm.
- All three reported per (query, arm), plus LLM cost (USD).

### 2.6 Config-space caveat (documented, deliberate)

The matrix swept only `executor_cores × executor_memory_gb × workers`
(15 distinct combos); it held **AQE = on** and never varied
`shuffle_partitions`, `advisory_partition_mb`, `broadcast_threshold_mb`,
`driver_memory_gb`. The Phase-2 free schema exposes all 8 knobs. Therefore:

- t* is a **floor over a subspace**, not a global optimum. If an arm beats t*, it
  found gains in knobs the grid never explored — reported as a finding.
- The Random control searches the **same free schema** as the LLM, so
  "LLM vs Random" is a fair within-space comparison; "arm vs t*" is the
  cross-reference to the dissertation.

---

## 3. Architecture

Reuse the existing closed loop end to end; add the four things it lacks for a
falsifiable free-schema campaign.

```
campaign YAML (mode: optimize, arm: llm|random, tries: k)
    │
    ├─ wiring.build_deps ── Deps(backend=SparkK8sBackend, guardian=Guardian(caps),
    │                            pack=free-range KnowledgePack, ...)
    │
    └─ runner._run_optimize_campaign
          for (dataset, query):
             loop.optimize(ep, deps, advisor, max_iters=B, patience, tries=k)
                 iter 0: DEFAULT_CFG                (baseline, k tries → median)
                 iter i: advisor.advise(history) → Guardian.validate
                          → backend.execute × k → median → feedback
             → journal opt_result {best_config, best_duration_ms, iterations, baseline_ms}

offline: analysis/oracle.py  (reads data/matrix/training_data.parquet)
         analysis/report.py   (joins journal opt_results with t* → regret/speedup/anytime)
```

### 3.1 New / changed components

1. **`loop.optimize(..., tries: int = 1)`** — run each proposed config `tries`
   times; feed the **median** success duration back (and track best on median).
   Also measure `DEFAULT_CFG` once as iteration 0 (median of k) to get
   `baseline_ms`. Returns additionally `baseline_ms` and per-iteration
   `samples`/`durations_ms`.
2. **`FreeRandomAdvisor`** (`advisor_loop/studies/advisors.py`) — samples each
   `ConfigVector` field uniformly within its Field range, rejection-sampling
   against a `Guardian` so proposals are feasible; deterministic per
   `(seed, dataset, query, len(history))`; never repeats a config within an
   episode. Same `advise(ep, context, probe, memories, history=[])` signature.
3. **Free-range pack** — replace the profile-menu framing. `packs/` presents the
   knob **ranges** (min–max, from `ConfigVector` Fields) + physical cluster caps
   + per-knob semantics (T1), and instructs "propose any values within these
   ranges that respect the caps; you are not limited to preset profiles."
   Remove/neutralize `executor-profiles.yaml`'s "choose one of these profiles"
   wording so the LLM is genuinely free.
4. **Advisor selection in the optimize campaign** — `_run_optimize_campaign`
   builds the advisor from `camp["arm"]` (`llm` → `Advisor`; `random` →
   `FreeRandomAdvisor`), instead of hardcoding `Advisor`. `tries` and the
   arm flow into `loop.optimize`.
5. **Offline analysis** — `analysis/oracle.py` (extract t* per (sf, query) from
   the consolidated parquet) and `analysis/report.py` (join journal
   `opt_result`s with t*, emit regret/speedup/anytime table + JSON + a
   `FINDINGS.md`).
6. **Config packs / datasets / campaign YAMLs** for the RNP cluster:
   real caps (see §3.2), `spark_config` (ssh_cmd, GHCR image, node_selector),
   S3/MinIO, and the smoke + campaign definitions.

### 3.2 Cluster caps (Guardian)

The Guardian must reflect real *schedulable* capacity, not the theoretical max
(other pods co-exist; contamination note in CLAUDE.md). Caps for this campaign:

```yaml
cluster:
  max_cores_per_container: 24      # per RNP quota
  max_memory_gb_per_container: 96
  max_pods: 40                     # RNP pod quota is 100; 40 leaves headroom and,
                                   # crucially, admits executor_instances=20 (20+1<40)
                                   # so the grid's q03 optimum (20 workers) is reachable
  total_cores: 120                 # conservative: leave headroom on the 200-core cluster
  total_memory_gb: 400
  storage: "MinIO S3; Delta"
```

The oracle's winning configs (72c/288GB × 6w for q23/q50; 40c/140GB × 20w for
q03) all fit: 20+1 pods < 40, 72 and 40 cores < 120, 288 and 140 GB < 400.

### 3.3 Isolation & safety

- **One run at a time** on the cluster (CLAUDE.md contamination rule). The
  campaign runner is serial by construction (loop → single submitter).
- **Zero touch to exp05 data.** The matrix parquet is read-only, copied once for
  analysis; no cluster benchmark is launched by the analysis path.
- **Resume-safe:** journal `episode_key` skips completed (dataset, query) per
  arm; incremental writes so a mid-campaign crash loses at most one proposal.
- **Cost guard:** LLM arm uses `deepseek/deepseek-v4-flash-0731`, reasoning off,
  `tool_choice=required`, `max_tokens=1024` (proven ~$0.002/12 calls in pilot).

---

## 4. Data flow (one (query, arm) episode)

1. `loop.optimize` measures `DEFAULT_CFG` (k tries) → `baseline_ms`.
2. Iteration i: `advisor.advise(ep, context, probe=None, mems, history)` →
   `Recommendation(config, rationale)`.
3. `Guardian.validate(config)` — infeasible → recorded, counts toward patience,
   re-propose.
4. `backend.execute(ep, rec)` k times → k `Outcome`s → median success duration +
   representative metrics.
5. Journal `opt_iter` (config, median duration, metrics, rationale); memory
   distill; history append (drives T3 telemetry feedback).
6. Stop at `patience` medians without improvement or at `max_iters = B`.
7. Journal `opt_result` (best_config, best_duration_ms, baseline_ms, iterations).
8. Offline: `report.py` joins with t* → regret/speedup/anytime + FINDINGS.

---

## 5. Testing

- **Unit (no cluster, no API):**
  - `FreeRandomAdvisor`: proposals within ranges; all feasible under a strict
    Guardian; deterministic; no repeats within an episode.
  - `loop.optimize(tries=k)`: with a fake backend returning scripted durations,
    asserts median is used for feedback/best and `baseline_ms` is the default's
    median; patience honored on medians.
  - `oracle.py`: on a tiny fixture parquet, t* = min success duration per
    (sf, query); ignores non-success.
  - `report.py`: regret/speedup math (incl. negative regret) on fixture
    opt_results + oracle.
  - Pack: rendered context contains ranges + caps and does **not** contain the
    "choose one of these profiles" constraint.
- **Integration (mocked submitter):** `_run_optimize_campaign` with `arm: random`
  and a fake `SparkSubmitter.runner` drives the loop end to end, writes journal.
- **Live (manual, gated):** smoke campaign (LLM, k=1, 3 queries) on RNP — the
  real validation; not in CI.

---

## 6. Deliverables

- Code: `tries` in `loop.optimize`; `FreeRandomAdvisor`; free-range pack;
  arm-select in `_run_optimize_campaign`; `analysis/oracle.py` + `report.py`.
- Config: `packs/cluster-rnp.yaml`, updated `datasets.yaml`,
  `campaigns/cluster-smoke-sf100.yaml`, `campaigns/cluster-sf100.yaml`.
- Data: `data/matrix/training_data.parquet` (+ README provenance).
- Tests as in §5.
- After the live smoke: `experiments/cluster-sf100/FINDINGS.md`.

---

## 7. Open questions (resolved)

- Goal: **B (falsifiable)**, smoke SF100 first. ✔
- Queries: **q03, q23, q50**. ✔
- Budget: **B=8, k=3** (smoke k=1). ✔
- Probe level: **L0**; DuckDB probe deferred to ablation. ✔
- Baseline: **matrix oracle** (primary) + **free Random-8** (control); BO
  deferred. ✔
- Config space: **(ii) free schema**; t* is a documented floor over a subspace. ✔
