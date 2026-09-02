# Research Direction — the path taken

This document records *which path we followed and why*, so the trajectory is
legible to future readers and to the ICDE EAB write-up. It is a decision log,
not a spec. Each phase points at the concrete artifacts it produced.

---

## Phase 1 (done) — Offline replay study, A0–A4

**Question.** Is an LLM config-advisor worth its cost for Spark executor
selection, versus trivial and classical baselines?

**Why replay first.** We already had a measured TPC-H result matrix (Exp05).
Replaying it (`ReplayBackend` over `experiments/06-tpch-matrix/reference.parquet`,
88 workloads, SF100–750) let us evaluate advisors **offline, with a
ground-truth oracle** (best measured config per workload → grid-relative
regret) and **zero cluster contact** — so the live Exp05 campaign was never
perturbed. Cheap, reproducible, falsifiable.

**Arms.** A0 fixed-best-on-average · A1 random search (8 iters) · A2 naive
cross-scale surrogate (a *simplified* LakeHelm proxy) · A4 LLM zero-shot ·
A3t LLM online, time-only feedback · A3m LLM online, +telemetry feedback.
Model: `deepseek/deepseek-v4-flash-0731` (reasoning off, `tool_choice=required`).

**Result** (full table + caveats in `studies_results/FINDINGS.md`):

| Arm | feas.% | regret mean | cost $ |
|-----|-------:|------------:|-------:|
| A0 fixed | 100.0 | 0.153 | 0.000 |
| A1 random-8 | 98.9 | **0.105** | 0.000 |
| A2 surrogate | 45.5 | 0.568 | 0.000 |
| A4 LLM zero-shot | 90.9 | 0.223 | 0.029 |
| A3t LLM online time-only | 90.9 | 0.185 | 0.072 |
| A3m LLM online +telemetry | 90.9 | 0.151 | 0.095 |

**Verdict.** Two real signals: (1) telemetry feedback helps the LLM
(A3t→A3m: −18% regret, −29% p90); (2) online iteration is monotone
(A4>A3t>A3m). But the headline is honest and unflattering: **the LLM does not
beat trivial baselines** — free random search (A1) dominates every LLM arm on
regret *and* feasibility, and A0 ties the best LLM at zero cost. Every LLM
miss (8/88) is an *infeasible* pick at SF≥250. For discrete selection over a
small, pre-measured menu, a frontier LLM advisor is not cost-competitive.

## Why Phase 1 is not the whole story

Two structural limits motivate the pivot:

1. **The replay only exercised 3 of 8 config knobs.** The matrix fixed
   `driver_memory`, `shuffle_partitions`, `advisory_partition`, `aqe_enabled`,
   `broadcast_threshold`; `ReplayBackend` matches on
   `(dataset, query, executor_cores, executor_memory_gb, executor_instances)`
   only. So the study measured selection over a 3-dimensional discrete menu,
   not real high-dimensional tuning.
2. **A fixed benchmarking matrix does not scale.** Enumerating the space costs
   **O(vᵏ)** — k knobs × v values each, × queries × tries. Every new knob
   multiplies collection cost. That is precisely the regime where a *pre-built
   matrix* becomes infeasible, and where an *adaptive* advisor should earn its
   keep — by navigating the space instead of enumerating it.

## Phase 2 (chosen next) — Blind online adaptive experimentation

**Thesis.** *Benchmarking matrices do not scale with configuration
dimensionality; an adaptive advisor with telemetry feedback finds good
configs under a fixed real-execution budget, regardless of k.*

**What changes vs Phase 1:**
- Swap `ReplayBackend` → `SparkK8sBackend` (real runs). The `loop.optimize`
  propose→execute→feedback→re-propose loop is unchanged — this is what it was
  built for.
- Drop the discrete 22-config menu from the advisor prompt: propose freely
  within the schema ranges (dynamic config, not fixed profiles).
- Optionally widen the schema with knobs we never tested (e.g. dynamic
  allocation, off-heap, shuffle service) — these already have a home:
  `spark_conf.py` already renders all 8 current knobs to `--conf`, and the
  `Guardian` already validates cluster-fit before a run is spent.

**Evaluation must change — we lose the oracle.** Blind, there is no measured
t*, so grid-relative regret is undefined. Replace it with:
- **speedup over the Spark default config** (relative improvement), and
- **anytime-best**: best runtime found by real-run k (a sample-efficiency curve).

**Non-negotiable control.** A non-LLM adaptive baseline under the *same*
real-run budget B — **random search** and **Bayesian optimization**
(Optuna/SMAC). Without it, "the LLM improved execution" is unfalsifiable
(random might too in B runs). The publishable claim is *sample efficiency*:
does the LLM reach a good config in fewer real runs than random / BO?

**Operational constraints:**
- Each proposal = one real Spark run (minutes–hours); OOM/timeout guarded by
  `Guardian` + `QUERY_TIMEOUT`; the run budget B must be an explicit cap.
- Runtime variance is real (co-execution contamination is documented) → decide
  tries-per-proposal or accept noisy feedback.
- **Must not touch the Exp05 cluster** — needs isolated capacity or a
  dedicated window.

**Minimal first experiment.** Few workloads (e.g. SF100 + SF500 × ~5 queries),
budget B ≈ 10–15 runs/workload, four arms under the same B: `default-config`,
`random-search`, `bayesian-opt`, `llm-online+telemetry`. Metric: speedup over
default + sample-efficiency curve.

## Open decisions (to close before building)

- Exact config schema for Phase 2 (which new knobs; continuous vs stepped ranges).
- Budget B and tries-per-proposal.
- Which BO implementation (Optuna TPE vs SMAC) as the classical baseline.
- Where to run in isolation without perturbing Exp05.
- How feedback noise is handled in the loop's stop criteria.
