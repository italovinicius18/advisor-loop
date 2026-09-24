"""A1 -- anchor via TypeSafe Jev (modelo de DECISAO tipada no OpenRouter).

Jev nao gera texto nem numeros: responde perguntas tipadas sobre um `state`
JSON -- `noul` (probabilidade de sim), `choice` (escolhe entre opcoes que NOS
definimos, com probabilidade por opcao), `score`. Logo ele nao pode propor
uma config; pode ESCOLHER entre candidatos gerados numericamente. E o extremo
mais barato/restrito da escada de ablacao do anchor (A0 heuristico -> A1 Jev
-> A2 Haiku -> A3 Sonnet): se um seletor sem geracao mantiver o desempenho do
hybrid3, o harness e a contribuicao e o modelo e um knob de custo.

Desenho por ancora (iteracao 0 e a cada llm_every):
  1. base = anchor heuristico (iteracao 0) ou melhor config ate agora;
  2. dois `noul` de diagnostico sobre o perfil/historico (I/O de shuffle domina?
     sub-provisionada?) -> orientam QUAIS candidatos gerar (consolidar vs
     espalhar, crescer vs encolher);
  3. 4-6 candidatos de forma (consolidacao x2/x0.5, instancias x1.5/x0.67,
     memoria x1.5), viaveis pelo Guardian e ineditos;
  4. um `choice`: "qual candidato minimiza o tempo desta query?" -> Jev devolve
     a opcao e as probabilidades; a escolhida vira a proposta. Tudo vai ao
     journal (kind=llm_call, com custo real de usage.cost).
API: POST https://openrouter.ai/api/alpha/decisions (chave OpenRouter).
"""
from __future__ import annotations

import json
import os
import time

import httpx
from dotenv import load_dotenv

from advisor_loop.advisor.heuristic import heuristic_anchor, parse_profile
from advisor_loop.advisor.hybrid import _success, consolidate, _scaled
from advisor_loop.episode import ConfigVector, Episode, Outcome, Recommendation

JEV_MODEL = "typesafe/jev-1.13"
_TRANSIENT = (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
              httpx.ReadTimeout, httpx.WriteError, httpx.PoolTimeout)


class JevClient:
    """Cliente minimo da Decisions API (mesma chave/credito do OpenRouter)."""

    def __init__(self, api_key: str | None = None,
                 base_url: str = "https://openrouter.ai/api/alpha",
                 transport: httpx.BaseTransport | None = None, max_retries: int = 4,
                 sleep=time.sleep, timeout: float = 60):
        load_dotenv()
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.max_retries, self._sleep = max_retries, sleep
        self.http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport,
                                 headers={"Authorization": f"Bearer {self.api_key}"})

    def decide(self, model: str, state: dict, questions: dict) -> dict:
        payload = {"model": model, "state": state, "questions": questions}
        for attempt in range(self.max_retries):
            try:
                r = self.http.post("/decisions", json=payload)
                break
            except _TRANSIENT:
                if attempt == self.max_retries - 1:
                    raise
                self._sleep(2 ** attempt)
        r.raise_for_status()
        return r.json()


def _shape(cv: ConfigVector) -> str:
    return f"{cv.executor_cores}c x {cv.executor_instances}w x {cv.executor_memory_gb}g"


def _describe(cv: ConfigVector) -> str:
    tc = cv.executor_cores * cv.executor_instances
    return (f"{cv.executor_instances} executors of {cv.executor_cores} cores and "
            f"{cv.executor_memory_gb} GB each ({tc} total cores, "
            f"{cv.executor_memory_gb / cv.executor_cores:.1f} GB per core)")


def build_candidates(base: ConfigVector, io_bound: bool, underprovisioned: bool,
                     guardian, seen: set[str], max_n: int = 6) -> list[ConfigVector]:
    """Candidatos de forma em torno de `base`, ordenados pelo que os diagnosticos
    sugerem primeiro; viaveis e ineditos; a propria base entra se inedita."""
    moves: list[ConfigVector] = []
    if io_bound:
        moves += [consolidate(base, 2.0), consolidate(base, 4.0)]
    else:
        moves += [consolidate(base, 0.5)]
    grow = base.model_copy(update={"executor_instances": _scaled(base.executor_instances, "executor_instances", 1.5)})
    shrink = base.model_copy(update={"executor_instances": _scaled(base.executor_instances, "executor_instances", 0.67)})
    moves += [grow, shrink] if underprovisioned else [shrink, grow]
    moves += [base.model_copy(update={"executor_memory_gb": _scaled(base.executor_memory_gb, "executor_memory_gb", 1.5)}),
              consolidate(base, 0.5) if io_bound else consolidate(base, 2.0)]
    out, hashes = [], set()
    for cand in [base] + moves:
        h = cand.config_hash()
        if h in hashes or h in seen or cand.executor_memory_gb > 64:
            continue
        if not guardian.validate(cand)[0]:
            continue
        hashes.add(h)
        out.append(cand)
        if len(out) >= max_n:
            break
    return out


def _state(profile_text: str, history) -> dict:
    succ = _success(history)
    rows = []
    for cv, qr in sorted(succ, key=lambda t: t[1].duration_ms):
        m = qr.metrics or {}
        rows.append({"config": _shape(cv), "runtime_s": round(qr.duration_ms / 1000, 1),
                     "cpu_util": round((m.get("executor_cpu_time_ms") or 0) /
                                       (cv.executor_cores * cv.executor_instances * qr.duration_ms), 2)
                     if qr.duration_ms else None,
                     "gc_s": round((m.get("gc_time_ms") or 0) / 1000, 1),
                     "shuffle_gb": round((m.get("shuffle_write_bytes") or 0) / 1024 ** 3, 2)})
    return {"query_profile": profile_text[-1500:], "measured_configs": rows[:8],
            "failed_configs": [_shape(cv) for cv, o in history
                               if o.status not in ("completed",)][:8]}


class JevAnchorAdvisor:
    def __init__(self, client: JevClient, journal, guardian, pinned: dict, model: str = JEV_MODEL):
        self.client, self.journal, self.guardian = client, journal, guardian
        self.pinned, self.model = dict(pinned), model
        self.envelope = None

    def set_envelope(self, env):
        self.envelope = env

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list[tuple[ConfigVector, Outcome]] = []) -> Recommendation:
        succ = _success(history)
        seen = {cv.config_hash() for cv, _ in history}
        prof = parse_profile(context)
        base = (min(succ, key=lambda t: t[1].duration_ms)[0] if succ
                else heuristic_anchor(prof, self.pinned, self.guardian))
        idx = context.find("## Perfil da query")
        state = _state(context[idx:] if idx >= 0 else "", history)
        diag = self.client.decide(self.model, state, {
            "io_bound": {"type": "noul",
                         "instructions": "Is this query's runtime dominated by shuffle/network I/O rather than CPU work?",
                         "criteria": {"true": "Large shuffle volume (>= 1 GB) and low CPU utilization: cores wait on data.",
                                      "false": "Small shuffle, or CPU utilization is the limiting factor."}},
            "underprovisioned": {"type": "noul",
                                 "instructions": "Would adding total cores (more executors) most likely reduce the runtime?",
                                 "criteria": {"true": "CPU work is large relative to total cores; utilization is high or runtime falls as cores grow.",
                                              "false": "CPU work is small; more cores only add scheduling overhead."}},
        })
        a = diag.get("answers", {})
        io_bound = float(a.get("io_bound", {}).get("noul", 0.5)) >= 0.5
        underprov = float(a.get("underprovisioned", {}).get("noul", 0.5)) >= 0.5
        cands = build_candidates(base, io_bound, underprov, self.guardian, seen)
        if self.envelope is not None:
            cands = [c for c in cands if self.envelope.check(c)[0]] or cands
        cost = float(diag.get("usage", {}).get("cost", 0.0) or 0.0)
        if not cands:
            self.journal.append({"kind": "llm_call", "model": self.model, "cost_usd": cost,
                                 "usage": diag.get("usage", {}), "jev": {"answers": a, "candidates": []}})
            return Recommendation(config=base, rationale="A1 jev: sem candidato inedito; repete base")
        if len(cands) == 1:
            pick, probs, conf = cands[0], {}, 1.0
        else:
            opts = {f"c{i}": _describe(c) for i, c in enumerate(cands)}
            ch = self.client.decide(self.model, {**state, "candidates": opts}, {
                "best": {"type": "choice",
                         "instructions": "Which candidate configuration will give the LOWEST runtime for this query on this cluster?",
                         "criteria": opts}})
            cost += float(ch.get("usage", {}).get("cost", 0.0) or 0.0)
            ans = ch.get("answers", {}).get("best", {})
            key = ans.get("choice") if ans.get("choice") in opts else "c0"
            pick = cands[int(key[1:])]
            probs, conf = ans.get("probabilities", {}), ans.get("confidence")
        self.journal.append({"kind": "llm_call", "model": self.model, "cost_usd": cost,
                             "usage": diag.get("usage", {}),
                             "jev": {"answers": {k: v.get("noul") for k, v in a.items()},
                                     "candidates": [_shape(c) for c in cands],
                                     "choice": _shape(pick), "probabilities": probs,
                                     "confidence": conf}})
        return Recommendation(config=pick, rationale=(
            f"A1 jev: io_bound={io_bound} underprovisioned={underprov} -> escolheu "
            f"{_shape(pick)} entre {len(cands)} candidatos (conf={conf})"))
