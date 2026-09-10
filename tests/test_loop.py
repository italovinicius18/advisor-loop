from advisor_loop.loop import optimize
from advisor_loop.episode import ConfigVector, Outcome, QueryResult, Episode, Recommendation
from advisor_loop.guardian import Guardian
from advisor_loop.journal import Journal
from advisor_loop.memory.store import MemoryStore

CLUSTER={"max_cores_per_container":24,"max_memory_gb_per_container":96,"max_pods":20,"total_cores":100,"total_memory_gb":400}
CFG=dict(executor_cores=8,executor_memory_gb=28,executor_instances=6,driver_memory_gb=8,shuffle_partitions=200,advisory_partition_mb=64,aqe_enabled=True,broadcast_threshold_mb=10)

class ScriptAdvisor:
    """devolve configs/durações roteirizadas por iteração."""
    def __init__(self, durations): self.durations=list(durations); self.i=0
    def advise(self, ep, ctx, probe, mems, history=None):
        self.i+=1
        return Recommendation(rationale="", config=ConfigVector(**{**CFG,"executor_instances":5+self.i}))

class ScriptBackend:
    def __init__(self, durations): self.durations=list(durations)
    def execute(self, ep, rec):
        d=self.durations.pop(0)
        if d is None: return Outcome(status="oom")
        return Outcome(status="completed", per_query={ep.queries[0]:QueryResult(status="success",duration_ms=d)}, wall_clock_s=d/1000)

class Pack:
    hash="h";
    def render(self,ep): return "ctx"

def _deps(tmp_path, backend):
    from advisor_loop.runner import Deps
    return Deps(client=None, journal=Journal(tmp_path/"j.jsonl"), pack=Pack(),
               guardian=Guardian(CLUSTER), backend=backend, memory=MemoryStore(tmp_path/"m.jsonl"),
               probe_factory=lambda ep: None)

def test_stops_on_no_improvement(tmp_path):
    ep=Episode(workload="w",dataset="sf1",queries=["q03"],provider="m")
    # 1a duracao = baseline (default cfg, iter 0); demais = propostas
    be=ScriptBackend([9000, 5000, 4000, 4200, 4300, 4100])
    r=optimize(ep, _deps(tmp_path,be), ScriptAdvisor([]), max_iters=5, patience=2)
    assert r["best_duration_ms"]==4000
    assert r["baseline_ms"]==9000
    assert len(r["iterations"])==4

def test_stops_on_max_iters(tmp_path):
    ep=Episode(workload="w",dataset="sf1",queries=["q03"],provider="m")
    be=ScriptBackend([9000, 5000,4900,4800,4700,4600])
    r=optimize(ep, _deps(tmp_path,be), ScriptAdvisor([]), max_iters=3, patience=2)
    assert len(r["iterations"])==3 and r["best_duration_ms"]==4800

class MetricsBackend:
    """Backend que devolve métricas reais por query (como o SparkK8sBackend)."""
    def execute(self, ep, rec):
        return Outcome(status="completed", wall_clock_s=20,
                       per_query={ep.queries[0]: QueryResult(
                           status="success", duration_ms=20000.0,
                           metrics={"shuffle_write_bytes": 116788, "gc_time_ms": 3776})})


def test_iteration_record_and_journal_include_metrics(tmp_path):
    import json
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    deps = _deps(tmp_path, MetricsBackend())
    r = optimize(ep, deps, ScriptAdvisor([]), max_iters=1, patience=2)
    assert r["iterations"][0]["metrics"]["shuffle_write_bytes"] == 116788
    ev = [json.loads(l) for l in open(tmp_path / "j.jsonl")]
    oi = [e for e in ev if e.get("kind") == "opt_iter"][0]
    assert oi["metrics"]["gc_time_ms"] == 3776 and oi["duration_ms"] == 20000.0


def test_oom_does_not_crash(tmp_path):
    ep=Episode(workload="w",dataset="sf1",queries=["q03"],provider="m")
    be=ScriptBackend([9000, None, 4000, 4100, 4200])
    r=optimize(ep, _deps(tmp_path,be), ScriptAdvisor([]), max_iters=4, patience=2)
    assert r["best_duration_ms"]==4000


class OneBadThenGoodAdvisor:
    """1a iteracao: config fora do cluster (Guardian rejeita); demais: config valida."""
    def __init__(self):
        self.i = 0
    def advise(self, ep, ctx, probe, mems, history=None):
        self.i += 1
        if self.i == 1:
            # 20 cores x 20 instances excede total_cores(100) e max_pods(20) do CLUSTER
            cfg = ConfigVector(**{**CFG, "executor_cores": 20, "executor_instances": 20})
        else:
            cfg = ConfigVector(**CFG)
        return Recommendation(rationale="", config=cfg)


def test_invalid_recommendation_does_not_crash_and_continues(tmp_path):
    """Guardian rejeita a 1a recomendacao (fora do cluster): o laco nao deve
    quebrar, deve registrar a iteracao como invalid_recommendation e seguir
    ate uma iteracao valida (Task 8 coverage gap)."""
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    be = ScriptBackend([9000, 4000, 3000])   # baseline + 2 execucoes validas
    r = optimize(ep, _deps(tmp_path, be), OneBadThenGoodAdvisor(), max_iters=3, patience=2)
    assert r["iterations"][0]["status"] == "invalid_recommendation"
    assert r["iterations"][0]["duration_ms"] is None
    assert len(r["iterations"]) == 3
    assert r["best_duration_ms"] == 3000


class ProbeCapturingAdvisor:
    """Registra o probe recebido em cada advise -> checa o gate de nivel."""
    def __init__(self):
        self.probes = []
    def advise(self, ep, ctx, probe, mems, history=None):
        self.probes.append(probe)
        return Recommendation(rationale="", config=ConfigVector(**CFG))


def _deps_probe(tmp_path, backend, sentinel):
    from advisor_loop.runner import Deps
    return Deps(client=None, journal=Journal(tmp_path / "j.jsonl"), pack=Pack(),
                guardian=Guardian(CLUSTER), backend=backend,
                memory=MemoryStore(tmp_path / "m.jsonl"),
                probe_factory=lambda ep: sentinel)


def test_probe_gated_off_at_L0(tmp_path):
    """Em L0 o laco passa probe=None ao advisor MESMO que o probe_factory
    devolva um probe (o datasets.yaml pode carregar probe_service): o nivel e
    autoritativo sobre a sondagem."""
    sentinel = object()
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m", level="L0")
    adv = ProbeCapturingAdvisor()
    optimize(ep, _deps_probe(tmp_path, ScriptBackend([9000, 4000]), sentinel),
             adv, max_iters=1, patience=2)
    assert adv.probes == [None]


def test_probe_passed_at_L1(tmp_path):
    """Em L1 o laco constroi o probe via probe_factory e o passa ao advisor."""
    sentinel = object()
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m", level="L1")
    adv = ProbeCapturingAdvisor()
    optimize(ep, _deps_probe(tmp_path, ScriptBackend([9000, 4000]), sentinel),
             adv, max_iters=1, patience=2)
    assert adv.probes == [sentinel]


class NBadThenGoodAdvisor:
    """As primeiras `bad` iteracoes propoem config inviavel; depois, validas."""
    def __init__(self, bad):
        self.bad, self.i = bad, 0
    def advise(self, ep, ctx, probe, mems, history=None):
        self.i += 1
        if self.i <= self.bad:
            cfg = ConfigVector(**{**CFG, "executor_cores": 20, "executor_instances": 20})
        else:
            cfg = ConfigVector(**{**CFG, "executor_instances": 5 + self.i})
        return Recommendation(rationale="", config=cfg)


def test_invalid_recommendations_do_not_trigger_early_stop(tmp_path):
    """Propostas inviaveis NAO devem contar no patience: com patience=2, duas
    inviaveis em sequencia abortariam o laco no comportamento antigo. Agora o
    laco segue ate uma valida e ate max_iters, provando que infeasible != 'sem
    melhora'."""
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    be = ScriptBackend([9000, 4000, 3000])   # baseline + 2 execucoes validas
    r = optimize(ep, _deps(tmp_path, be), NBadThenGoodAdvisor(bad=2),
                 max_iters=4, patience=2)
    assert [it["status"] for it in r["iterations"]] == [
        "invalid_recommendation", "invalid_recommendation", "completed", "completed"]
    assert r["best_duration_ms"] == 3000
