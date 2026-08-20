import json
from dataclasses import dataclass
from advisor_loop.episode import Decision
from advisor_loop.journal import Journal

@dataclass
class SupervisorPolicy:
    decision_interval_s: int = 600
    min_observation_s: int = 600
    max_wall_clock_s: int = 28800
    gc_ratio_trigger: float = 0.4
    stall_s: int = 600

_DECIDE = {"type": "function", "function": {
    "name": "submit_decision",
    "parameters": {"type": "object",
                   "required": ["action", "reason", "evidence", "confidence"],
                   "properties": {"action": {"enum": ["continue", "kill"]},
                                  "reason": {"type": "string"},
                                  "evidence": {"type": "array", "items": {"type": "string"}},
                                  "confidence": {"type": "number"}}}}}

class Supervisor:
    def __init__(self, client, journal: Journal, policy: SupervisorPolicy,
                 model: str = "openai/gpt-oss-20b"):
        self.client, self.journal, self.policy, self.model = client, journal, policy, model

    def _triggered(self, s: dict) -> list[str]:
        t = []
        if s["executors_lost"] > 0:
            t.append(f"{s['executors_lost']} executors perdidos")
        if s["last_progress_s"] > self.policy.stall_s:
            t.append(f"sem progresso ha {s['last_progress_s']}s")
        if s["run_time_ms"] and s["gc_time_ms"] / s["run_time_ms"] > self.policy.gc_ratio_trigger:
            t.append("GC acima do limiar")
        return t

    def check(self, snapshot: dict, elapsed_s: float) -> Decision:
        if elapsed_s > self.policy.max_wall_clock_s:
            d = Decision(action="kill", reason="hard cap de wall-clock (backstop de orcamento)",
                         evidence=[f"elapsed={elapsed_s}s"], confidence=1.0)
        elif elapsed_s < self.policy.min_observation_s:
            d = Decision(action="continue", reason="janela minima de observacao",
                         evidence=[], confidence=1.0)
        else:
            triggers = self._triggered(snapshot)
            if not triggers:
                d = Decision(action="continue", reason="sem gatilhos", evidence=[], confidence=1.0)
            else:
                r = self.client.chat(self.model, [
                    {"role": "system", "content":
                     "Decida continuar ou matar um job de query distribuida. "
                     "Responda SEMPRE via submit_decision."},
                    {"role": "user", "content":
                     f"Gatilhos: {triggers}\nSnapshot: {json.dumps(snapshot)}\n"
                     f"Tempo decorrido: {elapsed_s}s"}], tools=[_DECIDE])
                args = json.loads(r.tool_calls[0]["function"]["arguments"])
                d = Decision(**args)
        self.journal.append({"kind": "decision", **d.model_dump(), "elapsed_s": elapsed_s})
        return d
