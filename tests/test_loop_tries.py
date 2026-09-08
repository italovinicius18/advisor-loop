from advisor_loop.loop import optimize
from advisor_loop.episode import ConfigVector, Episode, Outcome, QueryResult, Recommendation

DEFAULT = dict(executor_cores=8, executor_memory_gb=8, executor_instances=1,
               driver_memory_gb=4, shuffle_partitions=200, advisory_partition_mb=64,
               aqe_enabled=True, broadcast_threshold_mb=10)


def _outcome(ms):
    return Outcome(status="completed",
                   per_query={"q": QueryResult(status="success", duration_ms=ms)})


class _ScriptedBackend:
    """Returns durations from a queue in order; records call order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def execute(self, ep, rec):
        ms = self.script.pop(0)
        self.calls.append((rec.config.config_hash(), ms))
        return _outcome(ms)


class _FixedAdvisor:
    def __init__(self, cfg):
        self.cfg = cfg

    def advise(self, ep, context, probe, mems, history=[]):
        return Recommendation(config=self.cfg, rationale="x")


class _Deps:
    class _J:
        def append(self, *a, **k):
            pass

    class _M:
        def retrieve(self, ep):
            return []

        def distill(self, *a, **k):
            pass

    class _P:
        def render(self, ep):
            return "ctx"

    class _G:
        def validate(self, cfg):
            return True, "ok"

    def __init__(self, backend):
        self.backend, self.journal, self.memory = backend, self._J(), self._M()
        self.pack, self.guardian = self._P(), self._G()
        self.probe_factory = lambda ep: None


def _ep():
    return Episode(workload="w", dataset="sf100", queries=["q"], provider="m")


def test_baseline_is_default_median():
    # iter0 default k=3 -> median(30,10,20)=20 ; iter1 proposal k=3 -> median(9,7,8)=8
    be = _ScriptedBackend([30, 10, 20, 9, 7, 8])
    cfg = ConfigVector(**{**DEFAULT, "shuffle_partitions": 400})
    res = optimize(_ep(), _Deps(be), _FixedAdvisor(cfg), max_iters=1, patience=1, tries=3)
    assert res["baseline_ms"] == 20
    assert res["best_duration_ms"] == 8
    assert res["iterations"][0]["duration_ms"] == 8
    assert res["iterations"][0]["durations_ms"] == [9, 7, 8]


def test_median_drives_best_tracking():
    # two proposals; medians 8 then 12 -> best stays 8
    be = _ScriptedBackend([100, 100, 100, 9, 7, 8, 13, 12, 11])

    class _TwoAdvisor:
        def __init__(self):
            self.n = 0

        def advise(self, ep, context, probe, mems, history=[]):
            self.n += 1
            sp = 300 if self.n == 1 else 500
            return Recommendation(
                config=ConfigVector(**{**DEFAULT, "shuffle_partitions": sp}), rationale="x")

    res = optimize(_ep(), _Deps(be), _TwoAdvisor(), max_iters=2, patience=2, tries=3)
    assert res["best_duration_ms"] == 8


def test_tries_default_one_is_backward_compatible():
    be = _ScriptedBackend([50, 40])   # iter0 default, iter1 proposal, single try each
    cfg = ConfigVector(**{**DEFAULT, "shuffle_partitions": 400})
    res = optimize(_ep(), _Deps(be), _FixedAdvisor(cfg), max_iters=1, patience=1)
    assert res["baseline_ms"] == 50
    assert res["best_duration_ms"] == 40
