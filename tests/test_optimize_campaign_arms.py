from advisor_loop.runner import _run_optimize_campaign, Deps
from advisor_loop.guardian import Guardian
from advisor_loop.episode import Outcome, QueryResult

CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}


class _BE:
    def __init__(self):
        self.calls = 0

    def execute(self, ep, rec):
        self.calls += 1
        return Outcome(status="completed",
                       per_query={"q23": QueryResult(status="success", duration_ms=100.0)})


class _J:
    def __init__(self):
        self.rows = []

    def append(self, r):
        self.rows.append(r)

    def has(self, k):
        return False


class _M:
    def retrieve(self, ep):
        return []

    def distill(self, *a, **k):
        pass


class _P:
    def render(self, ep):
        return "ctx"


def _deps(backend):
    return Deps(client=None, journal=_J(), pack=_P(), guardian=Guardian(CAPS),
                backend=backend, memory=_M(), probe_factory=lambda ep: None)


def test_random_arm_drives_loop():
    be = _BE()
    deps = _deps(be)
    camp = {"name": "t", "mode": "optimize", "arm": "random", "datasets": ["sf100"],
            "queries": ["q23"], "level": "L0", "provider": "m",
            "max_iters": 3, "patience": 2, "tries": 2}
    out = _run_optimize_campaign(camp, deps, None)  # adv unused for random
    assert out["episodes_run"] == 1
    opt = [r for r in deps.journal.rows if r.get("kind") == "opt_result"]
    assert len(opt) == 1
    assert opt[0]["arm"] == "random"
    assert opt[0]["baseline_ms"] == 100.0
    # tries=2 -> backend called 2x baseline + 2x per proposal
    assert be.calls >= 4
