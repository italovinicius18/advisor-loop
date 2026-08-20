import json, pytest
from advisor_loop.advisor.advisor import Advisor, _SUBMIT
from advisor_loop.advisor.provider import LLMResponse
from advisor_loop.episode import Episode
from advisor_loop.journal import Journal
from advisor_loop.probes.budget import ProbeBudgetExceeded

CFG = {"executor_cores": 4, "executor_memory_gb": 14, "executor_instances": 4,
       "driver_memory_gb": 8, "shuffle_partitions": 400, "advisory_partition_mb": 64,
       "aqe_enabled": True, "broadcast_threshold_mb": 10}

class FakeClient:
    def __init__(self, scripted):
        self.scripted = list(scripted)
    def chat(self, model, messages, tools=None, temperature=0.2):
        return self.scripted.pop(0)

class FakeProbe:
    """Probe fake que sempre levanta ProbeBudgetExceeded em table_stats."""
    def table_stats(self, table):
        raise ProbeBudgetExceeded("budget esgotado")

def _submit_call():
    args = json.dumps({**CFG, "rationale": "workload medio"})
    return LLMResponse(message={}, cost_usd=0.001, tool_calls=[
        {"id": "1", "function": {"name": "submit_recommendation", "arguments": args}}])

def _probe_call():
    return LLMResponse(message={}, cost_usd=0.0005, tool_calls=[
        {"id": "p1", "function": {"name": "table_stats",
                                   "arguments": json.dumps({"table": "store_sales"})}}])

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

def test_advise_handles_probe_error_and_still_submits(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    adv = Advisor(FakeClient([_probe_call(), _submit_call()]), j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    rec = adv.advise(ep, context="ctx", probe=FakeProbe(), memories=[])
    assert rec.config.executor_cores == 4
    probe_events = j.events("probe")
    assert len(probe_events) == 1
    assert "budget esgotado" in probe_events[0]["error"]


def _submit_call_out_of_bounds():
    """driver_memory_gb=99 viola le=16 de ConfigVector -> pydantic.ValidationError."""
    bad = {**CFG, "driver_memory_gb": 99}
    args = json.dumps({**bad, "rationale": "config invalida propositalmente"})
    return LLMResponse(message={}, cost_usd=0.001, tool_calls=[
        {"id": "1", "function": {"name": "submit_recommendation", "arguments": args}}])


def test_submit_schema_declares_bounds_for_driver_memory_gb():
    props = _SUBMIT["function"]["parameters"]["properties"]
    assert props["driver_memory_gb"].get("minimum") == 4
    assert props["driver_memory_gb"].get("maximum") == 16


def test_advise_retries_after_out_of_bound_submit(tmp_path):
    """Um ValidationError do pydantic nao deve propagar (crash da CLI) - o advisor
    deve devolver o erro ao modelo como tool result e deixar ele tentar de novo."""
    j = Journal(tmp_path / "j.jsonl")
    adv = Advisor(FakeClient([_submit_call_out_of_bounds(), _submit_call()]), j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    rec = adv.advise(ep, context="ctx", probe=None, memories=[])
    assert rec.config.driver_memory_gb == 8
    assert rec.config.executor_cores == 4
