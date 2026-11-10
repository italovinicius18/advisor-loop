import json
import re
import subprocess
import time

from advisor_loop.backends.launcher import render_launcher
from advisor_loop.episode import ConfigVector
from spark_advisor.driver import classify_error

RESULT_MARKER = "ADVISOR_RESULT "
_ERR_PAT = re.compile(r"(Exception|Error|OOM|OutOfMemory|Killed|exit code|ADVISOR_FATAL|ADVISOR_.*ERROR)")


def error_excerpt(logs: str, limit: int = 1500) -> str:
    """Linhas de erro dos logs (driver+launcher) em vez do cabecalho. Na rodada
    v3 quatro propostas 24c x 96g morreram em 'driver crash' e o journal so
    guardou os WARNs de boot do Spark -- impossivel diagnosticar depois."""
    hits = [ln.strip() for ln in logs.splitlines() if _ERR_PAT.search(ln)]
    text = "\n".join(hits[-15:]) if hits else logs[-limit:]
    return text[-limit:]

TERMINAL_PHASES = {"Succeeded", "Failed"}


class SparkSubmitter:
    """Submete um pod spark-submit via SSH->kubectl, aguarda término e coleta resultado.

    `runner` é injetável: recebe a lista de argumentos (self.ssh_cmd + [remote_cmd])
    e devolve um objeto com `.stdout`/`.returncode` (ex.: subprocess.CompletedProcess).
    """

    def __init__(self, ssh_cmd: list[str], image: str, result_bucket: str = "advisor-results",
                 poll_interval_s: float = 10, max_wall_clock_s: float = 1200,
                 node_selector: dict | None = None,
                 pod_template_file: str | None = None,
                 node_affinity_hostnames: list[str] | None = None,
                 runner=subprocess.run):
        self.ssh_cmd = ssh_cmd
        self.image = image
        self.result_bucket = result_bucket
        self.poll_interval_s = poll_interval_s
        self.max_wall_clock_s = max_wall_clock_s
        self.node_selector = node_selector
        self.pod_template_file = pod_template_file
        self.node_affinity_hostnames = node_affinity_hostnames
        self.runner = runner

    def _run(self, remote_cmd: str):
        # capture_output+text: sem isso o subprocess.run real devolve .stdout=None
        # e o parse (.strip()/logs) quebra — os testes usam um runner fake que ja
        # popula .stdout, entao so a execucao real expunha esse buraco.
        return self.runner(self.ssh_cmd + [remote_cmd], capture_output=True, text=True)

    def submit(self, config: ConfigVector, sf: str, query: str, run_id: str,
               tries: int = 1, wall_clock_s: float | None = None) -> dict:
        # wall_clock_s (v4): teto POR EXECUCAO relativo a baseline; None -> o teto
        # absoluto do submitter. Multiplicado por (tries+1) como antes.
        per_run = self.max_wall_clock_s if wall_clock_s is None else min(wall_clock_s,
                                                                         self.max_wall_clock_s)
        # deploy-mode cluster: o pod launcher roda spark-submit (bloqueante via
        # waitAppCompletion) e CRIA um pod driver separado. O marcador
        # ADVISOR_RESULT sai no stdout do DRIVER, nao do launcher.
        launcher_pod = f"advisor-{run_id}"
        driver_pod = f"advisor-{run_id}-driver"
        result_s3 = f"s3://{self.result_bucket}/{run_id}.json"
        manifest = render_launcher(config, sf, query, run_id, self.image, result_s3,
                                   node_selector=self.node_selector,
                                   pod_template_file=self.pod_template_file,
                                   node_affinity_hostnames=self.node_affinity_hostnames,
                                   tries=tries)

        try:
            apply_result = self._run(f"cat <<'EOF' | kubectl apply -f -\n{manifest}\nEOF")
            if apply_result.returncode != 0:
                return {"status": "failed", "duration_ms": None, "metrics": {},
                        "error": f"apply failed: {apply_result.stdout}"}

            # Com tries em sessao o pod roda (warmup + tries) execucoes: o teto
            # de wall clock escala junto, senao um baseline lento (q23 default
            # 1 executor: ~350s x 4) estoura os 1200s e vira timeout -> baseline
            # None (aconteceu no replay do oraculo).
            phase = self._poll(launcher_pod, per_run * max(1, tries + 1))

            if phase == "timeout":
                return {"status": "failed", "duration_ms": None, "metrics": {},
                        "error": f"timeout waiting for pod {launcher_pod} "
                                 f"(>{per_run * max(1, tries + 1):.0f}s)"}

            # logs do DRIVER carregam o ADVISOR_RESULT; fallback p/ launcher se o
            # driver nem chegou a ser criado (ex.: erro de submit/imagem no launcher).
            driver_logs = self._run(f"kubectl logs {driver_pod}").stdout or ""
            launcher_logs = self._run(f"kubectl logs {launcher_pod}").stdout or ""

            result = self._parse_result(driver_logs) or self._parse_result(launcher_logs)
            if result is not None:
                # status do proprio driver e autoritativo (success/oom/timeout/failed)
                return result

            # sem marcador: classificar pelo texto dos logs (driver + launcher)
            combined = driver_logs + "\n" + launcher_logs
            excerpt = error_excerpt(combined)
            if phase == "Succeeded":
                return {"status": classify_error(excerpt) if classify_error(excerpt) != "failed"
                        else "failed", "duration_ms": None, "metrics": {},
                        "error": "launcher Succeeded mas sem ADVISOR_RESULT (driver crash mascarado): "
                                 + excerpt}
            return {"status": classify_error(excerpt), "duration_ms": None, "metrics": {},
                    "error": excerpt}
        finally:
            self._run(f"kubectl delete pod {launcher_pod} {driver_pod} --ignore-not-found")

    def _poll(self, pod_name: str, budget_s: float | None = None) -> str:
        budget_s = self.max_wall_clock_s if budget_s is None else budget_s
        elapsed = 0.0
        while True:
            phase = self._run(
                f"kubectl get pod {pod_name} -o jsonpath='{{.status.phase}}'"
            ).stdout.strip()
            if phase in TERMINAL_PHASES:
                return phase
            if elapsed >= budget_s:
                return "timeout"
            time.sleep(self.poll_interval_s)
            elapsed += self.poll_interval_s

    def _parse_result(self, logs: str):
        for line in logs.splitlines():
            if line.startswith(RESULT_MARKER):
                return json.loads(line[len(RESULT_MARKER):])
        return None
