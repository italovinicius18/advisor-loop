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


def test_fixed_replay_ignores_envelope_and_can_skip_baseline():
    # Regua top-k: o replay mede a celula da MATRIZ como ela e. O envelope v4
    # (piso >=8g/executor) rejeitaria 2c x 16w x 7g; e o baseline default
    # nao serve a regua e domina o custo no sf500.
    be = _BE()
    deps = _deps(be)

    def _sizer(ep):
        raise AssertionError("arm fixed nao deve consultar o envelope")
    deps.sizer = _sizer
    cell = {"executor_cores": 2, "executor_memory_gb": 7, "executor_instances": 16,
            "driver_memory_gb": 8, "shuffle_partitions": 200, "advisory_partition_mb": 64,
            "aqe_enabled": True, "broadcast_threshold_mb": 10}
    camp = {"name": "t", "mode": "optimize", "arm": "fixed", "datasets": ["sf100"],
            "queries": ["q23"], "level": "L0", "provider": "oracle-replay",
            "max_iters": 1, "patience": 1, "tries": 3, "baseline": False,
            "configs": {"q23": cell}}
    _run_optimize_campaign(camp, deps, None)
    opt = [r for r in deps.journal.rows if r.get("kind") == "opt_result"][0]
    assert opt["baseline_ms"] is None and opt["baseline_error"] is None
    assert opt["best_duration_ms"] == 100.0
    assert opt["iterations"][0]["config"]["executor_memory_gb"] == 7
    assert be.calls == 3                       # so a celula, sem baseline


def test_campaign_can_disable_envelope():
    # Repeticao de uma campanha v3.1 (sem envelope) com o codigo atual: a
    # campanha desliga o envelope explicitamente, senao o advisor nao e o mesmo.
    be = _BE()
    deps = _deps(be)

    def _sizer(ep):
        raise AssertionError("envelope: false nao deve consultar o sizer")
    deps.sizer = _sizer
    camp = {"name": "t", "mode": "optimize", "arm": "random", "datasets": ["sf750"],
            "queries": ["q23"], "level": "L0", "provider": "m",
            "max_iters": 2, "patience": 2, "tries": 1, "envelope": False}
    out = _run_optimize_campaign(camp, deps, None)
    assert out["episodes_run"] == 1
    assert not [r for r in deps.journal.rows if r.get("kind") == "envelope"]


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
