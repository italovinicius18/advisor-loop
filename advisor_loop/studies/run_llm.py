"""Braços LLM do estudo de regret offline (A3/A4) — via API OpenRouter.

CUSTA $: cada workload dispara chamadas de API. Use --limit para smoke.

    python -m advisor_loop.studies.run_llm REFERENCE.parquet --model MODEL \
        [--arm a3m|a3t|a4] [--limit N] [--max-iters K] [--out results.json]

Braços (mesmo Advisor LLM real; muda o que ele vê):
  a3m — online, histórico COM telemetria (shuffle/gc/...)   [ablação: +telemetria]
  a3t — online, histórico SÓ-tempo (ReplayBackend emit_metrics=False)
  a4  — zero-shot: uma única proposta (max_iters=1, sem histórico)

Reusa a mesma malha offline do run_offline (ReplayBackend, regret vs oráculo);
só troca o advisor por Advisor(client) e compartilha o journal p/ somar custo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from advisor_loop.advisor.advisor import Advisor
from advisor_loop.advisor.provider import OpenRouterClient
from advisor_loop.backends.replay import ReplayBackend
from advisor_loop.studies.advisors import config_space
from advisor_loop.studies.regret import (
    AdvisorSpec, _CostJournal, _StudyPack, run_advisor, aggregate, workloads,
)


def _menu(reference) -> list[tuple[int, int, int]]:
    """As triplas (cores, mem_gb, instances) da grade — o menu discreto que o
    LLM DEVE respeitar para que a proposta case com uma célula medida."""
    return [(c.executor_cores, c.executor_memory_gb, c.executor_instances)
            for c in config_space(reference)]


# Cap de saída por chamada: a rationale é limitada a 120 palavras pelo system
# prompt; sem cap, um modelo de raciocínio pode gerar respostas longas que
# estagnam sobre streaming instável (causa raiz do travamento do 1º sweep).
_MAX_TOKENS = 1024


def build_arm(arm: str, reference, client, max_iters: int):
    """Retorna (spec, backend, journal) para o braço pedido.

    reasoning desligado + tool_choice="required": o modelo "flash" DEVE emitir
    uma tool call (não texto livre) e sem trace de raciocínio -> ~12s/chamada em
    vez de ~90s + resposta sem tool call (que estagnava a varredura em loop de
    12 rounds). É o modo "flash" fiel: rápido, direto ao submit_recommendation."""
    journal = _CostJournal()
    advisor = Advisor(client, journal, max_tokens=_MAX_TOKENS,
                      tool_choice="required", reasoning={"enabled": False})
    if arm == "a3m":
        backend = ReplayBackend(reference, emit_metrics=True)
        return AdvisorSpec("A3m-llm-online+telemetry", advisor, max_iters=max_iters), backend, journal
    if arm == "a3t":
        backend = ReplayBackend(reference, emit_metrics=False)
        return AdvisorSpec("A3t-llm-online-timeonly", advisor, max_iters=max_iters), backend, journal
    if arm == "a4":
        backend = ReplayBackend(reference, emit_metrics=True)
        return AdvisorSpec("A4-llm-zeroshot", advisor, max_iters=1), backend, journal
    raise SystemExit(f"arm inválido: {arm} (use a3m|a3t|a4)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="run_llm")
    ap.add_argument("reference")
    ap.add_argument("--model", required=True,
                    help="ex.: deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--arm", default="a3m", choices=["a3m", "a3t", "a4"])
    ap.add_argument("--limit", type=int, default=0, help="nº de workloads (0=todos)")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--benchmark", default="TPC-H", help="nome no prompt (TPC-H | TPC-DS)")
    ap.add_argument("--shard", default="", help="i/n: processa so os workloads com indice %% n == i")
    ap.add_argument("--resume", action="store_true",
                    help="retoma do --out existente: pula workloads ja gravados e soma o custo ja pago")
    args = ap.parse_args(argv)

    # timeout curto por request: um call que estagna falha rápido e é
    # re-tentado (backoff) em vez de bloquear ~120s -> varredura não trava.
    client = OpenRouterClient(timeout=60)
    spec, backend, journal = build_arm(args.arm, args.reference, client, args.max_iters)
    spec.provider = args.model

    wls = workloads(args.reference)
    if args.limit:
        wls = wls[:args.limit]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        wls = [w for k, w in enumerate(wls) if k % n == i]

    pack = _StudyPack(menu=_menu(args.reference), benchmark=args.benchmark)
    print(f"braço={spec.name} model={args.model} workloads={len(wls)} "
          f"max_iters={spec.max_iters} menu={len(pack.menu)} configs "
          f"max_tokens={_MAX_TOKENS}", flush=True)

    # --resume: um processo interrompido (sessao encerrada, queda de rede) nao
    # paga de novo os workloads ja feitos; o custo ja gasto entra no total.
    done: list = []
    prior_cost, prior_calls = 0.0, 0
    if args.resume and args.out and os.path.exists(args.out):
        (_, prev), = json.load(open(args.out)).items()
        done = prev["records"]
        prior_cost, prior_calls = prev["agg"]["cost_usd"], prev["agg"]["n_llm_calls"]
        have = {(r["dataset"], r["query"]) for r in done}
        wls = [w for w in wls if w not in have]
        print(f"retomando: {len(done)} workloads ja feitos, {len(wls)} restantes", flush=True)

    run_records: list = list(done)

    def _progress(idx, total, rec, cost):
        run_records.append(rec)
        r = (f"{rec['regret']:.3f}" if rec["regret"] is not None else "MISS")
        err = f" ERR={rec['error']}" if rec.get("error") else ""
        print(f"[{idx}/{total}] {rec['dataset']}/{rec['query']} regret={r} "
              f"iters={rec['iters']} cost=${cost:.4f}{err}", flush=True)
        # persistência incremental: se o processo morrer no meio, os workloads
        # já feitos ficam salvos (varredura longa sobre API instável).
        if args.out:
            partial = {"records": run_records, "cost_usd": prior_cost + journal.cost_usd,
                       "n_llm_calls": prior_calls + journal.n_llm}
            with open(args.out, "w") as fh:
                json.dump({spec.name: {"agg": aggregate(partial),
                          "records": run_records}}, fh, indent=2)

    run = run_advisor(spec, backend, wls, journal=journal, pack=pack,
                      progress=_progress)
    run = {"records": done + run["records"], "cost_usd": prior_cost + run["cost_usd"],
           "n_llm_calls": prior_calls + run["n_llm_calls"]}
    agg = aggregate(run)
    print(json.dumps(agg, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({spec.name: {"agg": agg, "records": run["records"]}}, fh, indent=2)
        print(f"resultados: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
