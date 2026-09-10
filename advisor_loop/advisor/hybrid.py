"""Braco HIBRIDO: o LLM ancora e diagnostica; uma busca numerica
deterministica refina.

Evidencia dos smokes r2-r6: o modelo diagnostica bem (viu extreme-fat ruim e
recuou) mas calibra mal os NUMEROS (super-provisiona a query leve,
sub-provisiona a pesada, satura caps). Padrao GPTuner/LLAMBO: o LLM escolhe a
regiao (forma dos executores, leitura de plano/GC/spill) e um otimizador
classico faz o refino numerico, que nao sabe qual benchmark esta rodando.

Politica (sem estado proprio -- tudo deriva do historico, logo reproduzivel e
retomavel):
  - iteracao 0 e a cada `llm_every`: LLM propoe (ancora / re-ancora vendo os
    resultados numericos no historico);
  - demais: PATTERN SEARCH em torno da melhor config ate agora, um eixo por
    vez (instancias, memoria, cores), passo relativo 50% -> 25% -> 12.5%,
    pulando configs ja vistas e inviaveis. A direcao inicial no eixo de
    paralelismo vem da UTILIZACAO medida (util<30% => menos primeiro;
    util>70% => mais primeiro) -- o sinal do driver dirige o numerico.
  - sem candidato inedito/viavel => devolve ao LLM.
"""
from __future__ import annotations

from advisor_loop.advisor.advisor import utilization
from advisor_loop.episode import ConfigVector, Episode, Outcome, Recommendation

AXES = ("executor_instances", "executor_memory_gb", "executor_cores")
RANGES = {"executor_instances": (1, 20), "executor_memory_gb": (4, 96),
          "executor_cores": (1, 24)}
STEPS = (0.5, 0.25, 0.125)


def _success(history):
    out = []
    for cv, o in history:
        qr = next((q for q in (o.per_query or {}).values()
                   if q.status == "success" and q.duration_ms is not None), None)
        if qr is not None:
            out.append((cv, qr))
    return out


def _scaled(value: int, axis: str, factor: float) -> int:
    lo, hi = RANGES[axis]
    new = int(round(value * factor))
    if new == value:                    # garante movimento (passo minimo = 1)
        new = value + (1 if factor > 1 else -1)
    return max(lo, min(hi, new))


class HybridAdvisor:
    def __init__(self, llm, guardian, llm_every: int = 3):
        if llm_every < 2:
            raise ValueError("llm_every deve ser >= 2 (senao nao ha passo numerico)")
        self.llm, self.guardian, self.llm_every = llm, guardian, llm_every

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list[tuple[ConfigVector, Outcome]] = []) -> Recommendation:
        succ = _success(history)
        n = len(history)
        if n == 0 or n % self.llm_every == 0 or not succ:
            return self.llm.advise(ep, context, probe, memories, history=history)
        step = self.numeric_step(history, succ)
        if step is None:
            return self.llm.advise(ep, context, probe, memories, history=history)
        return step

    def numeric_step(self, history, succ) -> Recommendation | None:
        best_cv, best_qr = min(succ, key=lambda t: t[1].duration_ms)
        seen = {cv.config_hash() for cv, _ in history}
        u = utilization(best_cv, best_qr.duration_ms, best_qr.metrics)
        # util baixa => encolher primeiro; alta ou desconhecida => crescer primeiro
        dirs = (-1, +1) if (u is not None and u < 0.3) else (+1, -1)
        for step in STEPS:
            for axis in AXES:
                for d in dirs:
                    cur = getattr(best_cv, axis)
                    new = _scaled(cur, axis, 1 + d * step)
                    if new == cur:
                        continue
                    cand = best_cv.model_copy(update={axis: new})
                    if cand.config_hash() in seen or not self.guardian.validate(cand)[0]:
                        continue
                    util_s = f", util={100 * u:.0f}%" if u is not None else ""
                    return Recommendation(
                        config=cand,
                        rationale=(f"hybrid pattern-search: {axis} {cur}->{new} "
                                   f"(passo {step:.0%}) em torno da melhor "
                                   f"{best_cv.executor_cores}c x {best_cv.executor_instances}w "
                                   f"x {best_cv.executor_memory_gb}g "
                                   f"({best_qr.duration_ms:g}ms{util_s})"))
        return None
