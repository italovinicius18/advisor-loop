# Offline regret study — A0–A4 (TPC-H matrix, replay)

**Setup.** 88 workloads `(dataset, query)` from the TPC-H reference matrix
(`experiments/06-tpch-matrix/reference.parquet`, SF100–750), each with an
oracle = best measured grid config. Regret `= t_A / t* − 1`, grid-relative.
Config space is the **discrete** menu of 22 executor-profile × worker combos
(the same measured cells; the LLM must pick from the menu). Everything runs
offline against `ReplayBackend` (no cluster contact — exp05 untouched).

**LLM.** `deepseek/deepseek-v4-flash-0731` via OpenRouter, `tool_choice=required`,
reasoning disabled (faithful "flash" mode: forces a tool call, ~12s/call
instead of a ~90s reasoning trace that never called the tool).

## Results

| Arm | feas.% | regret mean | median | p90 | cost $ | LLM calls |
|-----|-------:|------------:|-------:|----:|-------:|----------:|
| **A0** fixed-best-on-avg | 100.0 | 0.153 | 0.120 | 0.370 | 0.000 | 0 |
| **A1** random search (8 iters) | 98.9 | **0.105** | 0.062 | 0.265 | 0.000 | 0 |
| **A2** naive surrogate (cross-scale) | 45.5 | 0.568 | 0.397 | 1.104 | 0.000 | 0 |
| **A4** LLM zero-shot (1 shot) | 90.9 | 0.223 | 0.174 | 0.493 | 0.029 | 208 |
| **A3t** LLM online, time-only history | 90.9 | 0.185 | 0.151 | 0.468 | 0.072 | 680 |
| **A3m** LLM online, +telemetry history | 90.9 | 0.151 | 0.119 | 0.330 | 0.095 | 718 |

## Findings

1. **Telemetry ablation (A3t → A3m).** Feeding real telemetry
   (shuffle/spill/gc/…) into the LLM's feedback history — not just wall time —
   drops regret mean 0.185 → 0.151 (**−18%**) and p90 0.468 → 0.330 (**−29%**)
   for +$0.023. Telemetry demonstrably improves the online LLM advisor.

2. **Online iteration helps monotonically.** A4 zero-shot (0.223) → A3t online
   time-only (0.185) → A3m online +telemetry (0.151). More feedback → lower
   regret, but cost climbs (208 → 680 → 718 calls).

3. **The LLM does not beat trivial baselines.** A1 random search over the same
   discrete menu (8 iters, **free**, 98.9% feasible) beats *every* LLM arm on
   regret (0.105 vs best-LLM 0.151) and feasibility (98.9% vs 90.9%). A0
   fixed-best matches A3m on regret (0.153 vs 0.151) at zero cost and 100%
   feasibility. For this discrete config-selection task, a frontier LLM advisor
   — even with telemetry and online iteration — is not cost-competitive.

4. **Feasibility frontier.** All LLM arms miss exactly 8/88 workloads (0 hard
   errors); every miss is an *infeasible* pick concentrated at **SF ≥ 250**.
   The LLM lacks the calibration to avoid configs that don't complete at high
   scale — configs A0/A1 avoid by construction. This is the LLM's real
   weakness here, not raw regret.

## Caveats

- **A2 is a simplified LakeHelm proxy**, not their MoE surrogate. Its 45.5%
  feasibility reflects naive cross-scale config transfer collapsing on the
  scaling frontier; do not read it as LakeHelm's performance.
- Single seed, `temperature=0.2`; the specific 8 missed workloads vary per arm
  (stochastic), but the ~9% miss rate and its SF≥250 concentration are stable.
- Regret is grid-relative to the measured 22-cell menu, not a global optimum.

Reproduce: `python -m advisor_loop.studies.run_offline REF` (A0–A2) and
`python -m advisor_loop.studies.run_llm REF --model … --arm {a4,a3t,a3m}`.
