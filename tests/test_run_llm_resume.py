"""run_llm --resume: pula workloads ja gravados no --out e soma o custo ja pago."""
import json

import duckdb

from advisor_loop.studies import run_llm


class _FakeAdvisor:
    def advise(self, ep, context, probe, memories, history=[]):
        from advisor_loop.studies.advisors import make_config
        from advisor_loop.episode import Recommendation
        return Recommendation(config=make_config(4, 14, 4), rationale="x")


def test_resume_skips_done_workloads(tmp_path, monkeypatch):
    ref = tmp_path / "ref.parquet"
    duckdb.sql(f"""COPY (SELECT * FROM (VALUES
        ('sf1','q1',4,14,4,'success',100.0), ('sf1','q2',4,14,4,'success',200.0),
        ('sf1','q3',4,14,4,'success',300.0)) t(dataset,query,executor_cores,executor_memory_gb,
        executor_instances,status,duration_ms)) TO '{ref}' (FORMAT parquet)""")
    out = tmp_path / "out.json"
    prev = {"dataset": "sf1", "query": "q1", "best_ms": 100.0, "oracle_ms": 100.0, "regret": 0.0,
            "hit": True, "iters": 1}
    out.write_text(json.dumps({"A3m": {"agg": {"cost_usd": 0.5, "n_llm_calls": 7}, "records": [prev]}}))
    seen = []
    real_build = run_llm.build_arm

    def fake_build(arm, reference, client, max_iters):
        spec, backend, journal = real_build(arm, reference, client, max_iters)
        spec.advisor = _FakeAdvisor()
        return spec, backend, journal
    monkeypatch.setattr(run_llm, "build_arm", fake_build)
    monkeypatch.setattr(run_llm, "OpenRouterClient", lambda **k: None)
    orig = run_llm.run_advisor

    def spy(spec, backend, wls, **k):
        seen.extend(wls)
        return orig(spec, backend, wls, **k)
    monkeypatch.setattr(run_llm, "run_advisor", spy)
    run_llm.main([str(ref), "--model", "m", "--arm", "a3m", "--max-iters", "1", "--out", str(out), "--resume"])
    assert seen == [("sf1", "q2"), ("sf1", "q3")]
    (_, v), = json.loads(out.read_text()).items()
    assert [r["query"] for r in v["records"]] == ["q1", "q2", "q3"]
    assert v["agg"]["cost_usd"] == 0.5 and v["agg"]["n_llm_calls"] == 7
