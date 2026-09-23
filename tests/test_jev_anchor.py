"""A1: Jev escolhe entre candidatos numericos; nunca inventa uma config."""
import json
import httpx
from advisor_loop.advisor.advisor import PINNED_MATRIX, render_profile
from advisor_loop.advisor.jev import JevAnchorAdvisor, JevClient, build_candidates
from advisor_loop.episode import ConfigVector, Episode, Outcome, QueryResult
from advisor_loop.guardian import Guardian
from advisor_loop.journal import Journal
from advisor_loop.loop import DEFAULT_CFG

CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}
GB = 1024 ** 3


def _ctx(cpu_ms, shuffle):
    return "## pack\n\n" + render_profile(DEFAULT_CFG, 90000, {
        "executor_cpu_time_ms": cpu_ms, "shuffle_write_bytes": shuffle})


class FakeDecisions:
    """Transport httpx que simula a Decisions API; grava os pedidos."""
    def __init__(self, io_bound=0.9, underprov=0.2, pick=None):
        self.io_bound, self.underprov, self.pick, self.requests = io_bound, underprov, pick, []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        qs = body["questions"]
        answers = {}
        for name, q in qs.items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": self.io_bound if name == "io_bound" else self.underprov}
            elif q["type"] == "choice":
                opts = list(q["criteria"])
                choice = self.pick if self.pick in opts else opts[-1]
                answers[name] = {"type": "choice", "choice": choice, "confidence": 0.8,
                                 "probabilities": {o: (0.8 if o == choice else 0.2 / max(1, len(opts) - 1)) for o in opts}}
        return httpx.Response(200, json={"id": "x", "model": body["model"], "answers": answers,
                                         "usage": {"input_tokens": 400, "output_tokens": 10, "cost": 0.00002}})


def _client(fake):
    return JevClient(api_key="k", transport=httpx.MockTransport(fake.handler))


def test_build_candidates_feasible_unique_and_diagnosis_ordered():
    g = Guardian(CAPS)
    base = ConfigVector(executor_cores=4, executor_memory_gb=8, executor_instances=6, **PINNED_MATRIX)
    io = build_candidates(base, io_bound=True, underprovisioned=False, guardian=g, seen=set())
    cpu = build_candidates(base, io_bound=False, underprovisioned=True, guardian=g, seen=set())
    assert io[0].config_hash() == base.config_hash()
    assert io[1].executor_cores == 8 and io[1].executor_instances == 3        # consolidacao x2 primeiro
    assert cpu[1].executor_cores == 2                                        # espalha primeiro
    assert len({c.config_hash() for c in io}) == len(io)
    assert all(g.validate(c)[0] and c.executor_memory_gb <= 64 for c in io + cpu)
    # base ja vista -> nao volta
    assert base.config_hash() not in {c.config_hash() for c in
                                      build_candidates(base, True, False, g, {base.config_hash()})}


def test_jev_anchor_iteration0_uses_heuristic_base_and_choice(tmp_path):
    fake = FakeDecisions(io_bound=0.9, underprov=0.1, pick="c1")
    j = Journal(tmp_path / "j.jsonl")
    adv = JevAnchorAdvisor(_client(fake), j, Guardian(CAPS), PINNED_MATRIX)
    ep = Episode(workload="w", dataset="sf100", queries=["q50"], provider="jev")
    rec = adv.advise(ep, _ctx(520_000, int(8.8 * GB)), None, [], history=[])
    assert len(fake.requests) == 2                              # noul battery + choice
    assert set(fake.requests[0]["questions"]) == {"io_bound", "underprovisioned"}
    assert fake.requests[0]["state"]["query_profile"].startswith("## Perfil da query")
    assert fake.requests[1]["questions"]["best"]["type"] == "choice"
    assert "escolheu" in rec.rationale and rec.config.shuffle_partitions == 200
    ev = [json.loads(l) for l in open(tmp_path / "j.jsonl")]
    call = [e for e in ev if e["kind"] == "llm_call"][0]
    assert call["model"] == "typesafe/jev-1.13" and abs(call["cost_usd"] - 0.00004) < 1e-9
    assert call["jev"]["choice"] == "8c x 3w x 48g" or call["jev"]["choice"] in call["jev"]["candidates"]


def test_jev_reanchor_uses_best_so_far_and_skips_seen(tmp_path):
    fake = FakeDecisions(io_bound=0.9, underprov=0.9, pick="c0")
    adv = JevAnchorAdvisor(_client(fake), Journal(tmp_path / "j.jsonl"), Guardian(CAPS), PINNED_MATRIX)
    ep = Episode(workload="w", dataset="sf100", queries=["q50"], provider="jev")
    best = ConfigVector(executor_cores=8, executor_memory_gb=32, executor_instances=3, **PINNED_MATRIX)
    worse = ConfigVector(executor_cores=4, executor_memory_gb=16, executor_instances=6, **PINNED_MATRIX)
    def out(ms):
        return Outcome(status="completed", per_query={"q50": QueryResult(
            status="success", duration_ms=ms, metrics={"executor_cpu_time_ms": 500_000,
                                                       "shuffle_write_bytes": int(8.8 * GB)})})
    rec = adv.advise(ep, _ctx(520_000, int(8.8 * GB)), None, [], history=[(worse, out(200000)), (best, out(100000))])
    assert rec.config.config_hash() not in {best.config_hash(), worse.config_hash()}
    st = fake.requests[0]["state"]
    assert st["measured_configs"][0]["config"] == "8c x 3w x 32g"     # melhor primeiro
    assert st["measured_configs"][0]["runtime_s"] == 100.0
