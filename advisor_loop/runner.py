import argparse, sys
from dataclasses import dataclass
from pathlib import Path
import yaml
from advisor_loop.advisor.advisor import Advisor, AdvisorError
from advisor_loop.episode import Episode, episode_key


@dataclass
class Deps:
    client: object
    journal: object
    pack: object
    guardian: object
    backend: object
    memory: object
    probe_factory: object       # (episode) -> BudgetedProbe | None


def run_campaign(campaign_path: str | Path, deps: Deps) -> dict:
    camp = yaml.safe_load(Path(campaign_path).read_text())
    adv = Advisor(deps.client, deps.journal)
    if camp.get("mode") == "optimize":
        return _run_optimize_campaign(camp, deps, adv)
    ran = skipped = 0
    cost = 0.0
    for dataset in camp["datasets"]:
        for query in camp["queries"]:
            for try_n in range(1, camp.get("tries", 1) + 1):
                ep = Episode(workload=camp["name"], dataset=dataset, queries=[query],
                             objective=camp.get("objective", "min_time"),
                             level=camp["level"], provider=camp["provider"])
                ek = episode_key(camp["name"], dataset, query, try_n, ep.provider, ep.level)
                if deps.journal.has(ek):        # config-independent: nunca re-executa no resume
                    skipped += 1
                    continue
                probe = deps.probe_factory(ep) if camp["level"] in ("L1", "L2") else None
                mems = deps.memory.retrieve(ep) if camp.get("memory") == "on" else []
                context = deps.pack.render(ep)
                try:
                    rec = adv.advise(ep, context, probe, mems)
                except AdvisorError as e:
                    deps.journal.append({"kind": "episode_end", "task_key": ek,
                                         "status": "invalid_recommendation", "reason": str(e),
                                         "query": query, "dataset": dataset,
                                         "provider": ep.provider, "level": ep.level,
                                         "rationale": ""})
                    ran += 1
                    continue
                ok, reason = deps.guardian.validate(rec.config)
                if not ok:                              # 1 retry com feedback
                    try:
                        rec = adv.advise(ep, context + f"\n\nREJEITADA: {reason}", probe, mems)
                        ok, reason = deps.guardian.validate(rec.config)
                    except AdvisorError as e:
                        ok, reason = False, str(e)
                if not ok:
                    deps.journal.append({"kind": "episode_end", "task_key": ek,
                                         "status": "invalid_recommendation", "reason": reason,
                                         "query": query, "dataset": dataset,
                                         "provider": ep.provider, "level": ep.level,
                                         "rationale": rec.rationale})
                    ran += 1
                    continue
                out = deps.backend.execute(ep, rec)
                for q, qr in out.per_query.items():
                    deps.journal.append({"kind": "query_result", "task_key": ek, "query": q,
                                         **qr.model_dump()})
                deps.journal.append({"kind": "episode_end", "task_key": ek,
                                     "status": out.status, "pack_hash": deps.pack.hash,
                                     "config": rec.config.model_dump(),
                                     "config_hash": rec.config.config_hash(),
                                     "query": query, "dataset": dataset,
                                     "provider": ep.provider, "level": ep.level,
                                     "rationale": rec.rationale})
                if camp.get("memory") == "on":
                    deps.memory.distill(ep, rec, out)
                ran += 1
                cost += out.cost_usd
    return {"episodes_run": ran, "episodes_skipped": skipped, "cost_usd": cost}


def _run_optimize_campaign(camp: dict, deps: Deps, adv: Advisor) -> dict:
    """Modo optimize (spec §2.5): por (dataset,query) chama o laco fechado
    optimize() em vez do ciclo single-shot. Memoria e sempre usada dentro do
    laco (optimize() ja retrieve/distill incondicionalmente) - nao ha gate de
    camp["memory"] aqui, ao contrario do fluxo single-shot acima."""
    from advisor_loop.loop import optimize          # import local: evita ciclo runner<->loop
    arm = camp.get("arm", "llm")
    if arm == "random":                             # braco de controle: proposta livre aleatoria
        from advisor_loop.studies.advisors import FreeRandomAdvisor
        advisor = FreeRandomAdvisor(deps.guardian, seed=camp.get("seed", 0))
    elif arm == "llm":
        advisor = adv                               # braco principal: Advisor (LLM)
    else:
        raise ValueError(f"arm desconhecido: {arm!r} (use 'llm' ou 'random')")
    tries = camp.get("tries", 1)
    ran = skipped = 0
    cost = 0.0
    for dataset in camp["datasets"]:
        for query in camp["queries"]:
            ep = Episode(workload=camp["name"], dataset=dataset, queries=[query],
                         objective=camp.get("objective", "min_time"),
                         level=camp["level"], provider=camp["provider"])
            ek = episode_key(camp["name"], dataset, query, 1, ep.provider, ep.level)
            if deps.journal.has(ek):        # config-independent: nunca re-executa no resume
                skipped += 1
                continue
            result = optimize(ep, deps, advisor, max_iters=camp.get("max_iters", 4),
                              patience=camp.get("patience", 2), tries=tries)
            deps.journal.append({"kind": "opt_result", "task_key": ek, "arm": arm,
                                 "query": query, "dataset": dataset,
                                 "provider": ep.provider, "level": ep.level,
                                 **result})
            ran += 1
    return {"episodes_run": ran, "episodes_skipped": skipped, "cost_usd": cost}


def main() -> int:
    ap = argparse.ArgumentParser(prog="advisor-loop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("campaign")
    args = ap.parse_args()
    from advisor_loop.wiring import build_deps          # Task 13
    summary = run_campaign(args.campaign, build_deps(Path(args.campaign)))
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
