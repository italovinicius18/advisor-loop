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
from advisor_loop.episode import episode_key

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
    # Resume real: LLM nao deterministico pode nao repetir a config exata.
    # deps sem scripted responses -> se o runner chamar advise() de novo, estoura IndexError.
    s2 = run_campaign(camp, _deps(tmp_path, []))
    assert s2["episodes_skipped"] == 1 and s2["episodes_run"] == 0
    ek = episode_key("pilot", "sf100", "q1", 1, "m", "L0")
    ends = [e for e in Journal(tmp_path / "j.jsonl").events("episode_end") if e["task_key"] == ek]
    assert len(ends) == 1


def test_stubborn_advisor_does_not_crash_campaign(tmp_path):
    """Advisor que nunca chama submit_recommendation deve virar dado, nao crash."""
    camp = _campaign(tmp_path)
    empty = LLMResponse(message={"content": "hmm"}, tool_calls=[])
    summary = run_campaign(camp, _deps(tmp_path, [empty] * 12))      # esgota max_rounds sem submit
    assert summary["episodes_run"] == 1
    ek = episode_key("pilot", "sf100", "q1", 1, "m", "L0")
    ends = [e for e in Journal(tmp_path / "j.jsonl").events("episode_end") if e["task_key"] == ek]
    assert len(ends) == 1
    assert ends[0]["status"] == "invalid_recommendation"


def _campaign_with(tmp_path, name, provider, level, fname="c.yaml"):
    c = {"name": name, "level": level, "provider": provider, "datasets": ["sf100"],
         "queries": ["q1"], "objective": "min_time", "tries": 1, "memory": "off"}
    p = tmp_path / fname
    p.write_text(yaml.safe_dump(c))
    return p


def test_resume_does_not_skip_across_provider(tmp_path):
    """Regression: two campaigns with the same journal but different provider must NOT
    resume-skip each other's episodes (silent wrong-data risk)."""
    camp_deepseek = _campaign_with(tmp_path, "pilot", "deepseek", "L0", "c1.yaml")
    s1 = run_campaign(camp_deepseek, _deps(tmp_path, [_submit_call()]))
    assert s1["episodes_run"] == 1 and s1["episodes_skipped"] == 0

    camp_sonnet = _campaign_with(tmp_path, "pilot", "sonnet", "L0", "c2.yaml")
    s2 = run_campaign(camp_sonnet, _deps(tmp_path, [_submit_call()]))
    assert s2["episodes_run"] == 1, "different provider must run a fresh episode, not skip"
    assert s2["episodes_skipped"] == 0


def test_resume_does_not_skip_across_level(tmp_path):
    """Regression: same name/provider but different level (L0 vs L1) must not resume-skip."""
    camp_l0 = _campaign_with(tmp_path, "pilot", "m", "L0", "c1.yaml")
    s1 = run_campaign(camp_l0, _deps(tmp_path, [_submit_call()]))
    assert s1["episodes_run"] == 1

    camp_l1 = _campaign_with(tmp_path, "pilot", "m", "L1", "c2.yaml")
    s2 = run_campaign(camp_l1, _deps(tmp_path, [_submit_call()]))
    assert s2["episodes_run"] == 1, "different level must run a fresh episode, not skip"


def test_resume_still_skips_when_everything_identical(tmp_path):
    """Sanity: identical name/provider/level still resume-skips as before."""
    camp = _campaign(tmp_path)
    s1 = run_campaign(camp, _deps(tmp_path, [_submit_call()]))
    assert s1["episodes_run"] == 1
    s2 = run_campaign(camp, _deps(tmp_path, []))
    assert s2["episodes_skipped"] == 1 and s2["episodes_run"] == 0
