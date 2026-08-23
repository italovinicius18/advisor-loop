import uuid
from advisor_loop.episode import Episode, Outcome, QueryResult, Recommendation

# QueryResult.status (success|failed|timeout|killed|oom) e Outcome.status
# (completed|driver_crash|aborted_by_supervisor|invalid_recommendation|
# infeasible|error|proposed|oom) sao Literals distintos que so coincidem em
# "oom". Um QueryResult "failed"/"timeout"/"killed" precisa ser traduzido
# para um Outcome.status valido -- sem este mapa, Outcome(status="failed")
# estoura ValidationError e derruba a campanha inteira (nao capturado em
# loop.py).
_OUTCOME_STATUS = {"oom": "oom", "failed": "error", "timeout": "error", "killed": "error"}


class SparkK8sBackend:
    """Backend que executa queries no Kubernetes via SparkSubmitter e mapeia
    resultados para Outcome. Consumes SparkSubmitter.submit() e produz
    QueryResult + Outcome com status agregado e wall_clock_s somado."""

    def __init__(self, submitter, run_id_fn=None):
        self.submitter = submitter
        self.run_id_fn = run_id_fn or self._default_run_id_fn

    def _default_run_id_fn(self):
        """Gera run_id padrão a partir de uuid4."""
        return str(uuid.uuid4())[:8]

    def execute(self, ep: Episode, rec: Recommendation) -> Outcome:
        per_query = {}
        total_wall_clock_ms = 0.0
        outcome_status = "completed"

        for query in ep.queries:
            run_id = self.run_id_fn()
            result_dict = self.submitter.submit(rec.config, ep.dataset, query, run_id)

            # Map dict to QueryResult
            query_result = QueryResult(
                status=result_dict.get("status", "failed"),
                duration_ms=result_dict.get("duration_ms"),
                error=result_dict.get("error"),
                metrics=result_dict.get("metrics", {})
            )
            per_query[query] = query_result

            # Accumulate wall clock time (None -> 0)
            dur_ms = result_dict.get("duration_ms")
            if dur_ms is not None:
                total_wall_clock_ms += dur_ms

            # Track first non-success status
            if query_result.status != "success" and outcome_status == "completed":
                outcome_status = query_result.status

        wall_clock_s = total_wall_clock_ms / 1000.0

        # outcome_status carrega o QueryResult.status literal (success/...);
        # traduz para o Outcome.status literal antes de construir o modelo.
        if outcome_status == "completed":
            mapped_status = "completed"
        else:
            mapped_status = _OUTCOME_STATUS.get(outcome_status, "error")

        return Outcome(
            status=mapped_status,
            per_query=per_query,
            wall_clock_s=wall_clock_s,
            cost_usd=0.0
        )
