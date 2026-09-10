from analysis.report import regret, speedup, anytime, build_report


def test_math():
    assert abs(regret(120, 100) - 0.2) < 1e-9
    assert regret(80, 100) < 0                 # beats oracle -> negative
    assert abs(speedup(200, 100) - 2.0) < 1e-9


def test_anytime_monotone():
    iters = [{"duration_ms": 300}, {"duration_ms": 120},
             {"duration_ms": 150}, {"duration_ms": 90}]
    assert anytime(iters) == [300, 120, 120, 90]


def test_anytime_skips_none():
    # leading pre-success None dropped; later None carries best forward
    iters = [{"duration_ms": None}, {"duration_ms": 200}, {"duration_ms": None},
             {"duration_ms": 150}]
    assert anytime(iters) == [200, 200, 150]


def test_build_report_with_median_oracle():
    ors = {(100, "q23"): 100.0}
    ors_med = {(100, "q23"): 120.0}
    res = [{"dataset": "sf100", "query": "q23", "arm": "llm", "best_duration_ms": 120.0,
            "baseline_ms": 240.0, "iterations": [{"duration_ms": 120.0}]}]
    rep = build_report(res, ors, ors_med)
    row = rep["rows"][0]
    assert abs(row["regret"] - 0.2) < 1e-9          # vs min
    assert abs(row["regret_med"]) < 1e-9            # vs mediana: empatou
    assert row["t_star_median_ms"] == 120.0
    assert abs(rep["summary"]["llm"]["median_regret_med"]) < 1e-9
    # sem oraculo mediano: campos None, nada quebra
    rep2 = build_report(res, ors)
    assert rep2["rows"][0]["regret_med"] is None
    assert rep2["summary"]["llm"]["median_regret_med"] is None


def test_harness_oracle_from_replay_journal(tmp_path):
    import json
    from analysis.report import harness_oracle_table
    j = tmp_path / "replay.jsonl"
    j.write_text("\n".join(json.dumps(r) for r in [
        {"kind": "opt_iter", "iter": 1, "duration_ms": 42000.0},
        {"kind": "opt_result", "dataset": "sf100", "query": "q03", "best_duration_ms": 42000.0},
        {"kind": "opt_result", "dataset": "sf100", "query": "q23", "best_duration_ms": None},
    ]) + "\n")
    oh = harness_oracle_table(str(j))
    assert oh == {(100, "q03"): 42000.0}          # sem best -> ausente, nao zero
    res = [{"dataset": "sf100", "query": "q03", "arm": "llm", "best_duration_ms": 63000.0,
            "baseline_ms": 100000.0, "iterations": []}]
    rep = build_report(res, {(100, "q03"): 20700.0}, None, oh)
    row = rep["rows"][0]
    assert abs(row["regret"] - (63000 / 20700 - 1)) < 1e-9        # vs matriz: ~2.04
    assert abs(row["regret_harness"] - 0.5) < 1e-9                # vs harness: 0.5
    assert abs(rep["summary"]["llm"]["median_regret_harness"] - 0.5) < 1e-9


def test_build_report():
    ors = {(100, "q23"): 100.0}
    res = [{"dataset": "sf100", "query": "q23", "arm": "llm", "best_duration_ms": 120.0,
            "baseline_ms": 240.0, "iterations": [{"duration_ms": 120.0}]}]
    rep = build_report(res, ors)
    row = rep["rows"][0]
    assert abs(row["regret"] - 0.2) < 1e-9 and abs(row["speedup"] - 2.0) < 1e-9
    assert row["t_star_ms"] == 100.0
    assert "llm" in rep["summary"]
    assert abs(rep["summary"]["llm"]["median_regret"] - 0.2) < 1e-9
