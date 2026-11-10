"""SF>=750: baseline por campanha e pinagem por campanha (a matriz usou
sp=2000/driver 16g nesses SFs; o default Spark nao roda)."""
import json
from advisor_loop.advisor.advisor import Advisor, PINNED_MATRIX
from advisor_loop.advisor.provider import LLMResponse
from advisor_loop.episode import ConfigVector, Episode, Outcome, QueryResult, Recommendation
from advisor_loop.journal import Journal
from advisor_loop.loop import optimize, DEFAULT_CFG
from tests.test_loop_tries import _Deps

BIG_PIN = dict(driver_memory_gb=16, shuffle_partitions=2000)
BASE = dict(executor_cores=12, executor_memory_gb=48, executor_instances=2, driver_memory_gb=16,
            shuffle_partitions=2000, advisory_partition_mb=64, aqe_enabled=True,
            broadcast_threshold_mb=10)


class _Backend:
    def __init__(self):
        self.seen = []

    def execute(self, ep, rec):
        self.seen.append(rec.config)
        return Outcome(status="completed", per_query={ep.queries[0]: QueryResult(
            status="success", duration_ms=1000.0)})


class _Adv:
    def advise(self, ep, ctx, probe, mems, history=[]):
        return Recommendation(rationale="", config=ConfigVector(**BASE))


def test_optimize_uses_campaign_baseline_and_records_it():
    be = _Backend()
    ep = Episode(workload="w", dataset="sf750", queries=["q23"], provider="m")
    r = optimize(ep, _Deps(be), _Adv(), max_iters=1, patience=1,
                 baseline_cfg=ConfigVector(**BASE))
    assert be.seen[0].executor_cores == 12 and be.seen[0].shuffle_partitions == 2000
    assert r["baseline_config"]["driver_memory_gb"] == 16
    r2 = optimize(ep, _Deps(_Backend()), _Adv(), max_iters=1, patience=1)
    assert r2["baseline_config"] == DEFAULT_CFG.model_dump()


class _Client:
    def __init__(self):
        self.msgs = None

    def chat(self, model, messages, tools=None, **kw):
        self.msgs = messages
        args = json.dumps({"executor_cores": 12, "executor_memory_gb": 48,
                           "executor_instances": 4, "rationale": "x"})
        return LLMResponse(message={}, tool_calls=[
            {"id": "1", "function": {"name": "submit_recommendation", "arguments": args}}])


def test_advisor_pinned_override_applies_and_is_announced(tmp_path):
    cl = _Client()
    adv = Advisor(cl, Journal(tmp_path / "j.jsonl"), knobs="matrix3", pinned=BIG_PIN)
    rec = adv.advise(Episode(workload="w", dataset="sf750", queries=["q23"], provider="m"),
                     "ctx", None, [])
    assert rec.config.shuffle_partitions == 2000 and rec.config.driver_memory_gb == 16
    assert rec.config.advisory_partition_mb == PINNED_MATRIX["advisory_partition_mb"]
    note = cl.msgs[1]["content"]
    assert "shuffle_partitions=2000" in note and "driver_memory_gb=16" in note
