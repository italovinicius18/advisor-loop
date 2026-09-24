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
  - v3: em query SHUFFLE-HEAVY o primeiro movimento numerico e CONSOLIDACAO
    (trocar instancias por cores/executor a cores totais ~constantes, escalando
    a memoria para manter mem/core). Evidencia: dois seeds do random acharam
    q23/q50 em 3-4 executores x 23c (shuffle fica local); os bracos LLM nunca
    chegaram la porque o modelo ancora em 4-6c/exec e o eixo de cores era o
    ultimo explorado. Propostas do LLM que repetem config ja vista caem para o
    passo numerico (q23 v2 desperdicou uma iteracao repetindo 5c x 20w).
"""
from __future__ import annotations

from advisor_loop.advisor.advisor import utilization
from advisor_loop.episode import ConfigVector, Episode, Outcome, Recommendation

AXES = ("executor_instances", "executor_memory_gb", "executor_cores")
RANGES = {"executor_instances": (1, 20), "executor_memory_gb": (4, 96),
          "executor_cores": (1, 24)}
STEPS = (0.5, 0.25, 0.125)
# Teto da consolidacao: na rodada v3 TODAS as propostas com 96g/executor
# (24c x 1w x 96g, 24c x 2w x 96g) morreram em driver crash -- 4 iteracoes
# perdidas em 2 bracos. O Guardian aceita 96g (cap por container), mas na
# pratica o pod nao roda. Consolidar so ate 64g e >= 2 instancias.
CONSOLIDATE_MAX_MEM_GB = 64
CONSOLIDATE_MIN_INSTANCES = 2


def _success(history):
    out = []
    for cv, o in history:
        qr = next((q for q in (o.per_query or {}).values()
                   if q.status == "success" and q.duration_ms is not None), None)
        if qr is not None:
            out.append((cv, qr))
    return out


def consolidate(cv: ConfigVector, factor: float) -> ConfigVector:
    """factor>1: menos instancias, mais cores/exec (cores totais ~iguais);
    factor<1: o inverso. Memoria por executor escala com cores/exec para manter
    mem/core. Clampa nas faixas do ConfigVector."""
    lo_i, hi_i = RANGES["executor_instances"]
    lo_c, hi_c = RANGES["executor_cores"]
    lo_m, hi_m = RANGES["executor_memory_gb"]
    inst = max(lo_i, min(hi_i, int(round(cv.executor_instances / factor))))
    cores = max(lo_c, min(hi_c, int(round(cv.executor_cores * factor))))
    mem = max(lo_m, min(hi_m, int(round(cv.executor_memory_gb * cores / cv.executor_cores))))
    return cv.model_copy(update={"executor_instances": inst, "executor_cores": cores,
                                 "executor_memory_gb": mem})


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
        self.envelope = None

    def set_envelope(self, env):
        self.envelope = env
        if hasattr(self.llm, "set_envelope"):
            self.llm.set_envelope(env)

    def _feasible(self, cand: ConfigVector) -> bool:
        if not self.guardian.validate(cand)[0]:
            return False
        return self.envelope is None or self.envelope.check(cand)[0]

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list[tuple[ConfigVector, Outcome]] = []) -> Recommendation:
        succ = _success(history)
        n = len(history)
        if n == 0 or n % self.llm_every == 0 or not succ:
            rec = self.llm.advise(ep, context, probe, memories, history=history)
            seen = {cv.config_hash() for cv, _ in history}
            if rec.config.config_hash() in seen and succ:
                # o LLM repetiu uma config ja medida: nao gastar a iteracao
                step = self.numeric_step(history, succ)
                if step is not None:
                    step.rationale = "LLM repetiu config ja vista -> " + step.rationale
                    return step
            return rec
        step = self.numeric_step(history, succ)
        if step is None:
            return self.llm.advise(ep, context, probe, memories, history=history)
        return step

    def numeric_step(self, history, succ) -> Recommendation | None:
        best_cv, best_qr = min(succ, key=lambda t: t[1].duration_ms)
        seen = {cv.config_hash() for cv, _ in history}
        u = utilization(best_cv, best_qr.duration_ms, best_qr.metrics)
        m = best_qr.metrics or {}
        shuffle = max(m.get("shuffle_read_bytes") or 0, m.get("shuffle_write_bytes") or 0)
        heavy_shuffle = shuffle >= 1 * 1024 ** 3
        # Direcao inicial no eixo de paralelismo: shuffle pesado => I/O-bound =>
        # crescer primeiro (util baixa ai e espera de I/O, nao CPU ociosa);
        # shuffle leve + util baixa => encolher primeiro; senao crescer.
        if heavy_shuffle:
            dirs = (+1, -1)
        elif u is not None and u < 0.3:
            dirs = (-1, +1)
        else:
            dirs = (+1, -1)
        def _rec(cand, what):
            util_s = f", util={100 * u:.0f}%" if u is not None else ""
            util_s += ", shuffle-heavy" if heavy_shuffle else ""
            return Recommendation(
                config=cand,
                rationale=(f"hybrid pattern-search: {what} em torno da melhor "
                           f"{best_cv.executor_cores}c x {best_cv.executor_instances}w "
                           f"x {best_cv.executor_memory_gb}g "
                           f"({best_qr.duration_ms:g}ms{util_s})"))

        # v3: consolidacao PRIMEIRO em shuffle-heavy (x2, x4, depois x0.5)
        if heavy_shuffle:
            for f in (2.0, 4.0, 0.5):
                cand = consolidate(best_cv, f)
                if cand.config_hash() in seen or cand.config_hash() == best_cv.config_hash():
                    continue
                if f > 1 and (cand.executor_memory_gb > CONSOLIDATE_MAX_MEM_GB
                              or cand.executor_instances < CONSOLIDATE_MIN_INSTANCES):
                    continue
                if not self._feasible(cand):
                    continue
                return _rec(cand, f"consolidacao x{f:g}: {best_cv.executor_cores}c x "
                                  f"{best_cv.executor_instances}w -> {cand.executor_cores}c x "
                                  f"{cand.executor_instances}w x {cand.executor_memory_gb}g "
                                  f"(cores totais {best_cv.executor_cores * best_cv.executor_instances}"
                                  f"->{cand.executor_cores * cand.executor_instances})")
        for step in STEPS:
            for axis in AXES:
                for d in dirs:
                    cur = getattr(best_cv, axis)
                    new = _scaled(cur, axis, 1 + d * step)
                    if new == cur:
                        continue
                    cand = best_cv.model_copy(update={axis: new})
                    if cand.config_hash() in seen or not self._feasible(cand):
                        continue
                    return _rec(cand, f"{axis} {cur}->{new} (passo {step:.0%})")
        return None
