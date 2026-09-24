"""A0 -- anchor HEURISTICO, sem LLM: a ablacao decisiva do braco hibrido.

Pergunta: o LLM contribui algo alem do que o perfil da iteracao 0 + regras
simples ja dizem? Nos traços do hybrid3 as ancoras do Sonnet sao medianas
(sf250 q50: 4c x 6w x 12g = 349s; a busca numerica e que chegou a 104s). Se
este anchor determinístico + o MESMO pattern search empatar com o hybrid3, o
modelo e dispensavel (o harness e a contribuicao); se perder, o anchor exige
conhecimento de dominio que as regras nao capturam.

Regras (todas a partir do perfil medido na iteracao 0 -- baseline_cfg):
  - trabalho de CPU (core-s) do baseline -> cores totais alvo: o que faria a
    query durar ~TARGET_S se fosse perfeitamente paralela, limitado a
    [MIN_CORES, MAX_CORES]; sem perfil -> 24 cores.
  - shuffle >= 1GB (I/O-bound) -> comeca CONSOLIDADO: 12c/exec e 4g/core;
    caso contrario 4c/exec e 2g/core (muitos executores pequenos).
  - re-ancoragem (a cada llm_every): NAO ha LLM; devolve None e o HybridAdvisor
    segue com o passo numerico (a politica vira 100% pattern search apos a
    ancora). Knobs extras: os pinados da campanha (mesmos do hybrid3).
Interface identica ao Advisor (advise) para plugar no HybridAdvisor.
"""
from __future__ import annotations

import re

from advisor_loop.episode import ConfigVector, Episode, Outcome, Recommendation

TARGET_S = 30.0          # duracao "ideal" que dimensiona os cores totais
MIN_CORES, MAX_CORES = 8, 96
_CPU = re.compile(r"trabalho de CPU total: (\d+) core-segundos")
_SHUF = re.compile(r"shuffle=([\d.]+)(B|KB|MB|GB|TB)")
_MULT = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}


def parse_profile(context: str) -> dict:
    """Extrai do bloco 'Perfil da query (iteracao 0)' -- o mesmo texto que o LLM
    recebe -- o trabalho de CPU (core-s) e o shuffle (bytes). Sem perfil -> {}."""
    out: dict = {}
    m = _CPU.search(context or "")
    if m:
        out["cpu_core_s"] = int(m.group(1))
    m = _SHUF.search(context or "")
    if m:
        out["shuffle_bytes"] = float(m.group(1)) * _MULT[m.group(2)]
    return out


def heuristic_anchor(profile: dict, pinned: dict, guardian) -> ConfigVector:
    cpu = profile.get("cpu_core_s")
    shuffle = profile.get("shuffle_bytes", 0.0)
    if cpu:
        cores_tot = int(round(cpu / TARGET_S))
    else:
        cores_tot = 24
    cores_tot = max(MIN_CORES, min(MAX_CORES, cores_tot))
    if shuffle >= 1024 ** 3:                 # I/O-bound: poucos executores gordos
        cpe, mem_per_core = 12, 4
    else:                                    # CPU/latencia: muitos pequenos
        cpe, mem_per_core = 4, 2
    inst = max(2, min(20, int(round(cores_tot / cpe))))
    cand = ConfigVector(executor_cores=cpe, executor_memory_gb=min(96, cpe * mem_per_core),
                        executor_instances=inst, **pinned)
    # se o Guardian rejeitar (caps de cores/memoria totais), encolhe instancias
    while inst > 2 and not guardian.validate(cand)[0]:
        inst -= 1
        cand = cand.model_copy(update={"executor_instances": inst})
    return cand


class HeuristicAnchorAdvisor:
    """Drop-in para o `llm` do HybridAdvisor. `pinned` = 5 knobs extras fixos
    (PINNED_MATRIX ou os da campanha)."""

    def __init__(self, guardian, pinned: dict):
        self.guardian, self.pinned = guardian, dict(pinned)
        self.envelope = None

    def set_envelope(self, env):
        self.envelope = env

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list[tuple[ConfigVector, Outcome]] = []) -> Recommendation:
        prof = parse_profile(context)
        if self.envelope is not None:
            # v4: sem perfil (baseline falhou) o envelope ainda dimensiona; com
            # perfil, o envelope so impoe os pisos.
            if "cpu_core_s" not in prof:
                prof = {"cpu_core_s": self.envelope.target_cores * TARGET_S,
                        "shuffle_bytes": prof.get("shuffle_bytes",
                                                  self.envelope.data_gb * 0.1 * 1024 ** 3)}
        cfg = heuristic_anchor(prof, self.pinned, self.guardian)
        if self.envelope is not None and not self.envelope.check(cfg)[0]:
            cfg = self.envelope.baseline(self.pinned, self.guardian.cluster)
        why = (f"A0 heuristico: cpu={prof.get('cpu_core_s', '?')} core-s, "
               f"shuffle={prof.get('shuffle_bytes', 0) / 1024 ** 3:.1f}GB -> "
               f"{cfg.executor_cores}c x {cfg.executor_instances}w x {cfg.executor_memory_gb}g")
        return Recommendation(config=cfg, rationale=why)
