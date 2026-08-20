import json
from advisor_loop.supervisor import Supervisor, SupervisorPolicy
from advisor_loop.advisor.provider import LLMResponse
from advisor_loop.journal import Journal
from tests.test_advisor import FakeClient

SNAP_OK = {"stages_completed": 5, "stages_total": 10, "tasks_completed": 100,
           "gc_time_ms": 100, "run_time_ms": 10000, "executors_lost": 0, "last_progress_s": 5}
SNAP_STALL = {**SNAP_OK, "executors_lost": 2, "last_progress_s": 900}

def _kill_call():
    args = json.dumps({"action": "kill", "reason": "sem progresso",
                       "evidence": ["2 executors perdidos"], "confidence": 0.9})
    return LLMResponse(message={}, tool_calls=[
        {"id": "1", "function": {"name": "submit_decision", "arguments": args}}])

def test_heuristic_paths_no_llm(tmp_path):
    sup = Supervisor(FakeClient([]), Journal(tmp_path / "j.jsonl"), SupervisorPolicy())
    assert sup.check(SNAP_STALL, elapsed_s=60).action == "continue"     # janela minima
    assert sup.check(SNAP_OK, elapsed_s=5000).action == "continue"      # tudo normal
    assert sup.check(SNAP_OK, elapsed_s=99999).action == "kill"         # hard cap

def test_trigger_calls_llm(tmp_path):
    sup = Supervisor(FakeClient([_kill_call()]), Journal(tmp_path / "j.jsonl"), SupervisorPolicy())
    d = sup.check(SNAP_STALL, elapsed_s=5000)
    assert d.action == "kill" and d.confidence == 0.9
