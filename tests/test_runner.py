import json, yaml
from advisor_loop.runner import run_campaign, Deps
from advisor_loop.journal import Journal
from advisor_loop.guardian import Guardian
from advisor_loop.memory.store import MemoryStore
from advisor_loop.advisor.provider import LLMResponse
from tests.test_guardian import CLUSTER
from tests.test_advisor import _submit_call, FakeClient
from tests.test_replay import _ref
from advisor_loop.backends.replay import ReplayBackend

class Pack:
    hash = "deadbeefdeadbeef"
    def render(self, ep):
        return f"ctx {ep.dataset}"

def _campaign(tmp_path):
    c = {"name": "pilot", "level": "L0", "provider": "m", "datasets": ["sf100"],
         "queries": ["q1"], "objective": "min_time", "tries": 1, "memory": "off"}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c))
    return p

def _deps(tmp_path, scripted):
    return Deps(client=FakeClient(scripted), journal=Journal(tmp_path / "j.jsonl"),
                pack=Pack(), guardian=Guardian(CLUSTER),
                backend=ReplayBackend(_ref(tmp_path)),
                memory=MemoryStore(tmp_path / "m.jsonl"), probe_factory=lambda ep: None)

def test_run_and_resume(tmp_path):
    camp = _campaign(tmp_path)
    s1 = run_campaign(camp, _deps(tmp_path, [_submit_call()]))
    assert s1["episodes_run"] == 1
    s2 = run_campaign(camp, _deps(tmp_path, [_submit_call()]))       # re-advise, mesma chave -> pulado
    assert s2["episodes_skipped"] == 1 and s2["episodes_run"] == 0
