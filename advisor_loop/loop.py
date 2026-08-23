from advisor_loop.episode import Episode, Outcome
from advisor_loop.runner import Deps


def optimize(episode: Episode, deps: Deps, advisor, max_iters: int = 4, patience: int = 2) -> dict:
    """Laço fechado de otimização (spec §2.5): propose -> validate -> execute ->
    distill -> re-propose. Para em patience iterações sem melhora ou em max_iters."""
    history: list[tuple] = []
    best_config = None
    best_dur = None
    no_improve = 0
    iterations: list[dict] = []
    probe = deps.probe_factory(episode)

    for i in range(1, max_iters + 1):
        context = deps.pack.render(episode)
        mems = deps.memory.retrieve(episode) if hasattr(deps.memory, "retrieve") else []
        rec = advisor.advise(episode, context, probe, mems, history=history)

        ok, reason = deps.guardian.validate(rec.config)
        if not ok:
            out = Outcome(status="invalid_recommendation")
            history.append((rec.config, out))
            no_improve += 1
            iterations.append({"iter": i, "config": rec.config.model_dump(),
                               "status": out.status, "duration_ms": None})
            if no_improve >= patience or i == max_iters:
                break
            continue

        out = deps.backend.execute(episode, rec)

        dur = None
        metrics: dict = {}
        for qr in out.per_query.values():
            if qr.status == "success":
                dur = qr.duration_ms
                metrics = qr.metrics or {}
                break

        # métricas reais (shuffle/spill/gc/...) no journal e no resumo -> a
        # evidência mostra o que o advisor viu, não só o tempo.
        deps.journal.append({"kind": "opt_iter", "iter": i, "status": out.status,
                             "config": rec.config.model_dump(),
                             "config_hash": rec.config.config_hash(),
                             "duration_ms": dur, "metrics": metrics,
                             "rationale": rec.rationale})
        deps.memory.distill(episode, rec, out)

        history.append((rec.config, out))
        if dur is not None and (best_dur is None or dur < best_dur):
            best_config, best_dur = rec.config, dur
            no_improve = 0
        else:
            no_improve += 1

        iterations.append({"iter": i, "config": rec.config.model_dump(),
                           "status": out.status, "duration_ms": dur,
                           "metrics": metrics})

        if no_improve >= patience or i == max_iters:
            break

    return {"best_config": best_config.model_dump() if best_config else None,
            "best_duration_ms": best_dur,
            "iterations": iterations,
            "history_len": len(history)}
