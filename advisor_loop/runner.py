import argparse, sys
from dataclasses import dataclass
from pathlib import Path
import yaml
from advisor_loop.advisor.advisor import Advisor
from advisor_loop.episode import Episode, task_key


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
    ran = skipped = 0
    cost = 0.0
    for dataset in camp["datasets"]:
        for query in camp["queries"]:
            for try_n in range(1, camp.get("tries", 1) + 1):
                ep = Episode(workload=camp["name"], dataset=dataset, queries=[query],
                             objective=camp.get("objective", "min_time"),
                             level=camp["level"], provider=camp["provider"])
                probe = deps.probe_factory(ep) if camp["level"] in ("L1", "L2") else None
                mems = deps.memory.retrieve(ep) if camp.get("memory") == "on" else []
                context = deps.pack.render(ep)
                rec = adv.advise(ep, context, probe, mems)
                tk = task_key(ep.workload, dataset, query, rec.config.config_hash(), try_n)
                if deps.journal.has(tk):
                    skipped += 1
                    continue
                ok, reason = deps.guardian.validate(rec.config)
                if not ok:                              # 1 retry com feedback
                    rec = adv.advise(ep, context + f"\n\nREJEITADA: {reason}", probe, mems)
                    ok, reason = deps.guardian.validate(rec.config)
                if not ok:
                    deps.journal.append({"kind": "episode_end", "task_key": tk,
                                         "status": "invalid_recommendation", "reason": reason})
                    ran += 1
                    continue
                out = deps.backend.execute(ep, rec)
                for q, qr in out.per_query.items():
                    deps.journal.append({"kind": "query_result", "task_key": tk, "query": q,
                                         **qr.model_dump()})
                deps.journal.append({"kind": "episode_end", "task_key": tk,
                                     "status": out.status, "pack_hash": deps.pack.hash,
                                     "config": rec.config.model_dump()})
                if camp.get("memory") == "on":
                    deps.memory.distill(ep, rec, out)
                ran += 1
                cost += out.cost_usd
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
