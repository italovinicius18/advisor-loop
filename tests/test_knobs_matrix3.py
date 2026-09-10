"""Ablacao knobs='matrix3': o advisor (LLM e random) so move os 3 eixos da
grade offline; os outros 5 ficam pinados nos valores fixos da matriz."""
import json
import pytest
from advisor_loop.advisor.advisor import (Advisor, PINNED_MATRIX, SHAPE_KNOBS,
                                          _SUBMIT, _SUBMIT_MATRIX3)
from advisor_loop.advisor.provider import LLMResponse
from advisor_loop.episode import Episode
from advisor_loop.guardian import Guardian
from advisor_loop.journal import Journal
from advisor_loop.studies.advisors import FreeRandomAdvisor

CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}


class FakeClient:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.last_kwargs = None

    def chat(self, model, messages, tools=None, **kw):
        self.last_kwargs = {"messages": messages, "tools": tools}
        return self.scripted.pop(0)


def _ep():
    return Episode(workload="w", dataset="sf100", queries=["q03"], provider="m")


def _submit(args: dict):
    return LLMResponse(message={}, tool_calls=[
        {"id": "1", "function": {"name": "submit_recommendation",
                                 "arguments": json.dumps(args)}}])


def test_matrix3_schema_exposes_only_shape_knobs():
    props = _SUBMIT_MATRIX3["function"]["parameters"]["properties"]
    assert set(props) == set(SHAPE_KNOBS) | {"rationale"}
    # o schema livre continua com os 8 knobs
    assert set(_SUBMIT["function"]["parameters"]["properties"]) > set(props)


def test_matrix3_pins_extra_knobs_even_if_model_sends_them(tmp_path):
    client = FakeClient([_submit({"executor_cores": 4, "executor_memory_gb": 14,
                                  "executor_instances": 6, "shuffle_partitions": 2000,
                                  "rationale": "x"})])
    adv = Advisor(client, Journal(tmp_path / "j.jsonl"), knobs="matrix3")
    rec = adv.advise(_ep(), "ctx", None, [])
    assert rec.config.executor_cores == 4 and rec.config.executor_instances == 6
    for k, v in PINNED_MATRIX.items():
        assert getattr(rec.config, k) == v     # sp=2000 enviado foi sobrescrito p/ 200
    # a tool oferecida foi a restrita e o contexto avisa o que esta fixo
    assert client.last_kwargs["tools"][0] is _SUBMIT_MATRIX3
    assert "shuffle_partitions=200" in client.last_kwargs["messages"][1]["content"]


def test_free_is_default_and_unchanged(tmp_path):
    client = FakeClient([_submit({"executor_cores": 4, "executor_memory_gb": 14,
                                  "executor_instances": 6, "driver_memory_gb": 8,
                                  "shuffle_partitions": 2000, "advisory_partition_mb": 64,
                                  "aqe_enabled": True, "broadcast_threshold_mb": 10,
                                  "rationale": "x"})])
    adv = Advisor(client, Journal(tmp_path / "j.jsonl"))
    rec = adv.advise(_ep(), "ctx", None, [])
    assert rec.config.shuffle_partitions == 2000
    assert client.last_kwargs["tools"][0] is _SUBMIT
    assert "Knobs ajustaveis" not in client.last_kwargs["messages"][1]["content"]


def test_unknown_knobs_rejected(tmp_path):
    with pytest.raises(ValueError):
        Advisor(FakeClient([]), Journal(tmp_path / "j.jsonl"), knobs="bogus")


def test_free_random_matrix3_pins_and_stays_feasible():
    g = Guardian(CAPS)
    adv = FreeRandomAdvisor(g, seed=1, knobs="matrix3")
    hist = []
    for _ in range(6):
        rec = adv.advise(_ep(), "c", None, [], history=hist)
        assert g.validate(rec.config)[0]
        for k, v in PINNED_MATRIX.items():
            assert getattr(rec.config, k) == v
        hist.append((rec.config, None))
    assert len({c.config_hash() for c, _ in hist}) == 6   # sem repeticao
