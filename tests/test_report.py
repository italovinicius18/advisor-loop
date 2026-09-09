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
