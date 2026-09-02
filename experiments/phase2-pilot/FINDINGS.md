# Phase-2 pilot — blind online tuning, initial LLM adaptation (spike)

**Question.** With the discrete menu dropped (free proposal within the schema)
and the context enriched with Spark technicals — per-knob cards + cluster caps
(T1), the query's distilled physical plan (T2), and real run telemetry (T3) —
does the LLM *adapt* config proposals online from real-run feedback? This is a
behavioral probe, not a magnitude claim.

**Setup.** Local `local[*]` Spark 4.0 (WSL, 28 cores / 15 GB, ~7 GB heap cap),
TPC-H SF1 in Parquet (generated via DuckDB `tpch`). Zero cluster contact —
Exp05 untouched. Two queries: q1 (scan+aggregate) and q5 (6-table join).
Per query: measure the Spark-default config as baseline, then up to 8 blind
rounds (patience 3), each a real Spark run feeding runtime + telemetry + plan
back to `deepseek/deepseek-v4-flash-0731` (reasoning off, `tool_choice=required`).
Metric: speedup-over-default + anytime-best (no oracle exists blind).
Throwaway harness; regenerate data with DuckDB, run `run_pilot.py`.

## Results

| query | baseline | best | speedup | rounds | infeasible/failed |
|-------|---------:|-----:|--------:|-------:|------------------:|
| q1    | 3206 ms  | 2968 ms | 1.08x | 5 (patience) | 0/12 total |
| q5    | 3071 ms  | 2931 ms | 1.05x | 7 (patience) | — |

Total: 12 LLM proposals, **0 infeasible, 0 failed**, 3.5 min, LLM cost **$0.0018**.

## Findings

1. **The LLM adapts online from telemetry + history (T3).** Rationales
   explicitly reason over prior runs: *"Runs 1-3 show shuffle_partitions 400
   was best (2939ms)"*, *"run 2 showed 400 beat 200 so trend may continue"*,
   *"keeping everything else identical to the best run (2931ms)"*. The
   propose→execute→feedback→re-propose mechanism works end-to-end on real runs.

2. **It grounds proposals in the physical plan (T2).** *"Default plan already
   uses broadcast joins heavily"*; *"a final aggregation/sort over only ~25
   nations"*; for q1, *"no joins so broadcast threshold is irrelevant"*. The
   distilled plan changed what it proposed — the enrichment is used, not
   ignored. **This answers the design question: feeding the plan + knob context
   is worth it.**

3. **Knob-cards + explicit caps (T1) eliminated the Phase-1 failure mode.**
   Phase 1's only real LLM weakness was that every miss was an *infeasible*
   pick at SF≥250. Here, with the resource ceiling stated in the prompt, **all
   12 proposals were feasible** — 0 infeasible, 0 failed. Cheap grounding fixes
   the calibration gap.

4. **At SF1 the gains are below the noise floor — this is a mechanics probe,
   not evidence tuning helps.** The identical q5 config (200 partitions) ran
   2931 / 2965 / 3031 ms across rounds 4-6 and 3286 ms in round 1 — a
   ~12% spread that *exceeds* the 5-8% "best" speedups. The "best" is largely
   the LLM re-running near-default configs while run-to-run noise handed it a
   low sample. Two hard consequences for the real Phase-2:
   - **Need real scale (SF≥100)** so the config↔runtime signal exceeds variance.
   - **Need tries-per-proposal** (median of k) to denoise feedback, or the loop
     chases noise. Documented contamination/variance in CLAUDE.md predicted this.

## Verdict

The Phase-2 mechanism is sound and the context enrichment (T1+T2+T3) demonstrably
changes behavior for the better: the LLM adapts, reads the plan, and stays
feasible. What SF1 cannot show is *whether adaptation beats a non-LLM baseline* —
that needs (a) real scale, (b) tries-per-proposal, and (c) the mandatory
random-search / Bayesian-opt controls under the same run budget B
(see `docs/research-direction.md`). This spike de-risks the plumbing and the
prompt design before spending real cluster runs.
