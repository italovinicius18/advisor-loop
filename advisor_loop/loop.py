import statistics
from advisor_loop.episode import ConfigVector, Episode, Outcome, Recommendation
from advisor_loop.runner import Deps


# Config Spark-default como ConfigVector: medida como iteracao 0 (baseline) para
# ancorar speedup-sobre-default sem depender de nenhum oraculo externo.
DEFAULT_CFG = ConfigVector(executor_cores=8, executor_memory_gb=8, executor_instances=1,
                           driver_memory_gb=4, shuffle_partitions=200, advisory_partition_mb=64,
                           aqe_enabled=True, broadcast_threshold_mb=10)


def _run_median(deps: Deps, ep: Episode, rec: Recommendation, tries: int):
    """Executa a config `tries` vezes e devolve (mediana_ms, duracoes, metricas).
    As metricas representativas vem da execucao cuja duracao esta mais proxima da
    mediana -> denoise do ruido run-to-run (que dominava o piloto SF1)."""
    durs: list[float] = []
    metrics_by: dict[float, dict] = {}
    for _ in range(tries):
        out = deps.backend.execute(ep, rec)
        for qr in out.per_query.values():
            if qr.status == "success" and qr.duration_ms is not None:
                durs.append(qr.duration_ms)
                metrics_by[qr.duration_ms] = qr.metrics or {}
                break
    if not durs:
        return None, [], {}
    med = statistics.median(durs)
    key = min(metrics_by, key=lambda d: abs(d - med))
    return med, durs, metrics_by[key]


def optimize(episode: Episode, deps: Deps, advisor, max_iters: int = 4,
             patience: int = 2, tries: int = 1) -> dict:
    """Laço fechado de otimização (spec §2.5): propose -> validate -> execute ->
    distill -> re-propose. Para em patience iterações sem melhora ou em max_iters.

    Cada proposta e executada `tries` vezes; a MEDIANA realimenta o advisor e
    dirige o best-tracking. A config default e medida como iteracao 0
    (baseline_ms) para ancorar o speedup-sobre-default."""
    history: list[tuple] = []
    best_config = None
    best_dur = None
    no_improve = 0
    iterations: list[dict] = []
    probe = deps.probe_factory(episode)

    # iteracao 0: baseline default (k tries -> mediana), sem entrar no historico
    baseline_ms, _bdurs, _bm = _run_median(
        deps, episode, Recommendation(config=DEFAULT_CFG, rationale="baseline"), tries)

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
                               "status": out.status, "duration_ms": None,
                               "durations_ms": [], "samples": 0})
            if no_improve >= patience or i == max_iters:
                break
            continue

        dur, durations, metrics = _run_median(deps, episode, rec, tries)
        status = "completed" if dur is not None else "error"
        # Outcome sintetico para memoria/historico: carrega a mediana + metricas
        # representativas (mesma forma que o backend produziria numa unica run).
        from advisor_loop.episode import QueryResult
        qr_status = "success" if dur is not None else "failed"
        out = Outcome(status=status, per_query={
            episode.queries[0]: QueryResult(status=qr_status, duration_ms=dur, metrics=metrics)},
            wall_clock_s=sum(durations) / 1000.0 if durations else 0.0)

        # métricas reais (shuffle/spill/gc/...) no journal e no resumo -> a
        # evidência mostra o que o advisor viu, não só o tempo.
        deps.journal.append({"kind": "opt_iter", "iter": i, "status": out.status,
                             "config": rec.config.model_dump(),
                             "config_hash": rec.config.config_hash(),
                             "duration_ms": dur, "durations_ms": durations,
                             "metrics": metrics, "rationale": rec.rationale})
        deps.memory.distill(episode, rec, out)

        history.append((rec.config, out))
        if dur is not None and (best_dur is None or dur < best_dur):
            best_config, best_dur = rec.config, dur
            no_improve = 0
        else:
            no_improve += 1

        iterations.append({"iter": i, "config": rec.config.model_dump(),
                           "status": out.status, "duration_ms": dur,
                           "durations_ms": durations, "samples": len(durations),
                           "metrics": metrics})

        if no_improve >= patience or i == max_iters:
            break

    return {"best_config": best_config.model_dump() if best_config else None,
            "best_duration_ms": best_dur,
            "baseline_ms": baseline_ms,
            "iterations": iterations,
            "history_len": len(history)}
