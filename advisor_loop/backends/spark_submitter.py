import json
import subprocess
import time

from advisor_loop.backends.launcher import render_launcher
from advisor_loop.episode import ConfigVector
from spark_advisor.driver import classify_error

RESULT_MARKER = "ADVISOR_RESULT "

TERMINAL_PHASES = {"Succeeded", "Failed"}


class SparkSubmitter:
    """Submete um pod spark-submit via SSH->kubectl, aguarda término e coleta resultado.

    `runner` é injetável: recebe a lista de argumentos (self.ssh_cmd + [remote_cmd])
    e devolve um objeto com `.stdout`/`.returncode` (ex.: subprocess.CompletedProcess).
    """

    def __init__(self, ssh_cmd: list[str], image: str, result_bucket: str = "advisor-results",
                 poll_interval_s: float = 10, max_wall_clock_s: float = 1200,
                 runner=subprocess.run):
        self.ssh_cmd = ssh_cmd
        self.image = image
        self.result_bucket = result_bucket
        self.poll_interval_s = poll_interval_s
        self.max_wall_clock_s = max_wall_clock_s
        self.runner = runner

    def _run(self, remote_cmd: str):
        return self.runner(self.ssh_cmd + [remote_cmd])

    def submit(self, config: ConfigVector, sf: str, query: str, run_id: str) -> dict:
        pod_name = f"advisor-{run_id}"
        result_s3 = f"s3://{self.result_bucket}/{run_id}.json"
        manifest = render_launcher(config, sf, query, run_id, self.image, result_s3)

        try:
            apply_result = self._run(f"cat <<'EOF' | kubectl apply -f -\n{manifest}\nEOF")
            if apply_result.returncode != 0:
                return {"status": "failed", "duration_ms": None, "metrics": {},
                        "error": f"apply failed: {apply_result.stdout}"}

            phase = self._poll(pod_name)

            if phase == "timeout":
                return {"status": "failed", "duration_ms": None, "metrics": {},
                        "error": f"timeout waiting for pod {pod_name} (>{self.max_wall_clock_s}s)"}

            logs = self._run(f"kubectl logs {pod_name}").stdout

            if phase == "Succeeded":
                result = self._parse_result(logs)
                if result is not None:
                    return result
                # TODO: fallback via `mc cat` do MinIO (result_s3) antes de desistir —
                # o driver grava em ambos (Task 2), mas logs bastam para o primeiro corte.
                return {"status": "failed", "duration_ms": None, "metrics": {},
                        "error": "pod Succeeded but no ADVISOR_RESULT marker in logs (driver crash mascarado)"}

            # phase == "Failed"
            return {"status": classify_error(logs), "duration_ms": None, "metrics": {},
                    "error": logs[:500]}
        finally:
            self._run(f"kubectl delete pod {pod_name}")

    def _poll(self, pod_name: str) -> str:
        elapsed = 0.0
        while True:
            phase = self._run(
                f"kubectl get pod {pod_name} -o jsonpath='{{.status.phase}}'"
            ).stdout.strip()
            if phase in TERMINAL_PHASES:
                return phase
            if elapsed >= self.max_wall_clock_s:
                return "timeout"
            time.sleep(self.poll_interval_s)
            elapsed += self.poll_interval_s

    def _parse_result(self, logs: str):
        for line in logs.splitlines():
            if line.startswith(RESULT_MARKER):
                return json.loads(line[len(RESULT_MARKER):])
        return None
