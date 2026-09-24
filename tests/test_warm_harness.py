"""Harness quente: tries em sessao no driver, TRIES/WARMUP no launcher, backend
mapeia durations/cold, loop usa 1 submissao por config, arm=fixed p/ replay."""
import yaml
from advisor_loop.backends.launcher import render_launcher
from advisor_loop.backends.spark_k8s import SparkK8sBackend
from advisor_loop.episode import ConfigVector, Episode, Outcome, QueryResult, Recommendation
from advisor_loop.loop import _run_median, optimize
from advisor_loop.studies.advisors import PerQueryFixedAdvisor
from spark_advisor.driver import pick_median_run
from tests.test_loop_tries import _Deps, _FixedAdvisor

CFG = dict(executor_cores=8, executor_memory_gb=28, executor_instances=6, driver_memory_gb=8,
           shuffle_partitions=200, advisory_partition_mb=64, aqe_enabled=True,
           broadcast_threshold_mb=10)


def test_pick_median_run_returns_median_and_matching_metrics():
    med, m = pick_median_run([30.0, 10.0, 20.0], [{"r": 1}, {"r": 2}, {"r": 3}])
    assert med == 20.0 and m == {"r": 3}
    med, m = pick_median_run([10.0, 30.0], [{"r": 1}, {"r": 2}])
    assert med == 20.0 and m == {"r": 1}            # empate: primeiro mais proximo
    assert pick_median_run([], []) == (None, {})


def test_launcher_passes_tries_and_warmup_to_driver():
    m = render_launcher(ConfigVector(**CFG), sf="sf100", query="q03", run_id="r1",
                        image="img", result_s3="s3://b/r1.json", tries=3, warmup=1)
    doc = yaml.safe_load(m)
    args = " ".join(doc["spec"]["containers"][0]["args"])
    assert "spark.kubernetes.driverEnv.TRIES=3" in args
    assert "spark.kubernetes.driverEnv.WARMUP=1" in args
    env = {e["name"]: e["value"] for e in doc["spec"]["containers"][0]["env"]}
    assert env["TRIES"] == "3" and env["WARMUP"] == "1"
    m1 = render_launcher(ConfigVector(**CFG), sf="sf100", query="q03", run_id="r1",
                         image="img", result_s3="s3://b/r1.json")
    assert "driverEnv.TRIES=1" in " ".join(yaml.safe_load(m1)["spec"]["containers"][0]["args"])


class _Sub:
    def __init__(self, result):
        self.result, self.seen = result, []

    def submit(self, config, sf, query, run_id, tries=1, wall_clock_s=None):
        self.seen.append(tries)
        return self.result


def test_backend_forwards_tries_and_maps_durations_and_cold():
    sub = _Sub({"status": "success", "duration_ms": 25000.0, "metrics": {"input_bytes": 1},
                "durations_ms": [26000.0, 25000.0, 24000.0], "cold_ms": 70000.0, "error": None})
    be = SparkK8sBackend(sub, run_id_fn=lambda: "rid")
    assert be.in_session_tries is True
    out = be.execute(Episode(workload="w", dataset="sf100", queries=["q03"], provider="m"),
                     Recommendation(rationale="", config=ConfigVector(**CFG)), tries=3)
    qr = out.per_query["q03"]
    assert sub.seen == [3]
    assert qr.duration_ms == 25000.0 and qr.durations_ms == [26000.0, 25000.0, 24000.0]
    assert qr.cold_ms == 70000.0


class _InSessionBackend:
    in_session_tries = True

    def __init__(self):
        self.calls = []

    def execute(self, ep, rec, tries=1, **kw):
        self.calls.append(tries)
        return Outcome(status="completed", per_query={ep.queries[0]: QueryResult(
            status="success", duration_ms=25000.0, durations_ms=[26000.0, 25000.0, 24000.0],
            cold_ms=70000.0, metrics={"gc_time_ms": 5})})


def test_loop_submits_once_per_config_when_backend_runs_tries_in_session():
    be = _InSessionBackend()
    ep = Episode(workload="w", dataset="sf100", queries=["q03"], provider="m")
    med, durs, m, err = _run_median(_Deps(be), ep, Recommendation(
        rationale="", config=ConfigVector(**CFG)), tries=3)
    assert be.calls == [3]                       # UMA submissao, nao tres
    assert med == 25000.0 and durs == [26000.0, 25000.0, 24000.0] and err is None
    assert m["cold_ms"] == 70000.0 and m["gc_time_ms"] == 5
    res = optimize(ep, _Deps(be), _FixedAdvisor(ConfigVector(**CFG)), max_iters=1,
                   patience=1, tries=3)
    assert be.calls == [3, 3, 3] and res["best_duration_ms"] == 25000.0   # baseline + 1 proposta


def test_in_session_failure_reports_error():
    class _Bad(_InSessionBackend):
        def execute(self, ep, rec, tries=1, **kw):
            return Outcome(status="oom", per_query={ep.queries[0]: QueryResult(
                status="oom", error="heap")})
    ep = Episode(workload="w", dataset="sf100", queries=["q03"], provider="m")
    med, durs, m, err = _run_median(_Deps(_Bad()), ep, Recommendation(
        rationale="", config=ConfigVector(**CFG)), tries=3)
    assert med is None and durs == [] and err == {"query_status": "oom", "error": "heap"}


def test_per_query_fixed_advisor_replays_oracle_cells():
    adv = PerQueryFixedAdvisor({"q03": {**CFG, "executor_cores": 2, "executor_instances": 20,
                                        "executor_memory_gb": 7}}, default=CFG)
    r = adv.advise(Episode(workload="w", dataset="sf100", queries=["q03"], provider="m"),
                   "c", None, [])
    assert (r.config.executor_cores, r.config.executor_instances) == (2, 20)
    r2 = adv.advise(Episode(workload="w", dataset="sf100", queries=["q99"], provider="m"),
                    "c", None, [])
    assert r2.config.executor_cores == 8                     # fallback default
    camp = yaml.safe_load(open("campaigns/cluster-oracle-replay-sf100.yaml"))
    assert camp["arm"] == "fixed" and set(camp["configs"]) == {"q03", "q23", "q50"}
    for c in camp["configs"].values():
        ConfigVector(**c)                                     # todas validas
