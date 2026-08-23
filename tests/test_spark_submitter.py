import json
from advisor_loop.backends.spark_submitter import SparkSubmitter
from tests.test_spark_conf import _cv

class FakeRunner:
    """Simula ssh→kubectl: guarda comandos, devolve saídas roteirizadas por etapa."""
    def __init__(self, phases, logs):
        self.phases = list(phases); self.logs = logs; self.calls = []
    def __call__(self, cmd, **kw):
        import types
        self.calls.append(cmd)
        joined = " ".join(cmd)
        out = ""
        if "apply" in joined: out = "pod/advisor-r1 created"
        elif "jsonpath" in joined or "get pod" in joined: out = self.phases.pop(0)
        elif "logs" in joined: out = self.logs
        elif "delete" in joined: out = "deleted"
        return types.SimpleNamespace(stdout=out, returncode=0)

def test_submit_success_reads_result():
    logs = "some spark noise\nADVISOR_RESULT " + json.dumps(
        {"status":"success","duration_ms":4300.0,"metrics":{"input_bytes":9},"error":None}) + "\nmore"
    r = FakeRunner(phases=["Pending","Running","Succeeded"], logs=logs)
    s = SparkSubmitter(ssh_cmd=["ssh","host"], image="img", runner=r, poll_interval_s=0)
    res = s.submit(_cv(), "sf1", "q03", "r1")
    assert res["status"] == "success" and res["duration_ms"] == 4300.0
    assert any("delete" in " ".join(c) for c in r.calls)   # limpeza chamada

def test_succeeded_without_result_is_failed():
    r = FakeRunner(phases=["Succeeded"], logs="no marker here")
    s = SparkSubmitter(ssh_cmd=["ssh","host"], image="img", runner=r, poll_interval_s=0)
    assert s.submit(_cv(), "sf1", "q03", "r2")["status"] == "failed"

def test_pod_failed_is_failed():
    r = FakeRunner(phases=["Running","Failed"], logs="java.lang.OutOfMemoryError")
    s = SparkSubmitter(ssh_cmd=["ssh","host"], image="img", runner=r, poll_interval_s=0)
    res = s.submit(_cv(), "sf1", "q03", "r3")
    assert res["status"] in ("failed","oom")
