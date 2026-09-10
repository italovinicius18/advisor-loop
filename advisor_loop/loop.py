import statistics
from advisor_loop.advisor.advisor import render_profile
from advisor_loop.episode import ConfigVector, Episode, Outcome, Recommendation
from advisor_loop.runner import Deps


# Config Spark-default como ConfigVector: medida como iteracao 0 (baseline) para
# ancorar speedup-sobre-default sem depender de nenhum oraculo externo.
DEFAULT_CFG = ConfigVector(executor_cores=8, executor_memory_gb=8, executor_instances=1,
                           driver_memory_gb=4, shuffle_partitions=200, advisory_partition_mb=64,
                           aqe_enabled=True, broadcast_threshold_mb=10)


def _run_median(deps: Deps, ep: Episode, rec: Recommendation, tries: int):
    """Executa a config `tries` vezes e devolve (mediana_ms, duracoes, metricas, err).
    As metricas representativas vem da execucao cuja duracao esta mais proxima da
    mediana -> denoise do ruido run-to-run (que dominava o piloto SF1).

    `err` = None quando ao menos uma try teve sucesso; caso contrario carrega
    {"query_status", "error"} da ultima execucao sem sucesso, para o journal
    registrar POR QUE a proposta falhou (timeout/oom/erro de query) em vez de um
    "error" mudo -- sem isso, q23/q50 no smoke gravaram error sem diagnostico."""
    durs: list[float] = []
    metrics_by: dict[float, dict] = {}
    err: dict | None = None
    for _ in range(tries):
        out = deps.backend.execute(ep, rec)
        for qr in out.per_query.values():
            if qr.status == "success" and qr.duration_ms is not None:
                durs.append(qr.duration_ms)
                metrics_by[qr.duration_ms] = qr.metrics or {}
                break
            err = {"query_status": qr.status, "error": qr.error}
    if not durs:
        return None, [], {}, err
    med = statistics.median(durs)
    key = min(metrics_by, key=lambda d: abs(d - med))
    return med, durs, metrics_by[key], None


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
    # O nivel da campanha e autoritativo sobre a sondagem (igual ao caminho
    # single-shot em runner.py): so L1/L2 recebem probe. Sem este gate, o
    # probe_factory seria chamado incondicionalmente e bastaria o bloco
    # probe_service no datasets.yaml para LIGAR a sondagem em qualquer campanha
    # optimize -- poluindo o contraste cientifico L0 (r6) vs L1.
    probe = deps.probe_factory(episode) if episode.level in ("L1", "L2") else None

    # iteracao 0: baseline default (k tries -> mediana), sem entrar no historico
    baseline_ms, _bdurs, baseline_metrics, _berr = _run_median(
        deps, episode, Recommendation(config=DEFAULT_CFG, rationale="baseline"), tries)
    # O baseline ja custou k execucoes: aproveita a telemetria dele como PERFIL
    # da query (trabalho de CPU, utilizacao, volume, plano) no contexto de todas
    # as iteracoes -- o peso real da query antes da primeira proposta.
    profile = (render_profile(DEFAULT_CFG, baseline_ms, baseline_metrics)
               if baseline_ms is not None else "")

    for i in range(1, max_iters + 1):
        context = deps.pack.render(episode)
        if profile:
            context = f"{context}\n\n{profile}"
        mems = deps.memory.retrieve(episode) if hasattr(deps.memory, "retrieve") else []
        rec = advisor.advise(episode, context, probe, mems, history=history)

        ok, reason = deps.guardian.validate(rec.config)
        if not ok:
            # Uma proposta inviavel NAO conta como "sem melhora": e uma rodada
            # desperdicada, nao um sinal de convergencia. Contar no patience
            # (como antes) fazia um unico deslize infeasible perto do fim
            # abortar a exploracao. Segue no historico (o LLM ve que foi
            # rejeitada) e consome uma iteracao (guard de max_iters), mas nao
            # avanca no_improve.
            out = Outcome(status="invalid_recommendation")
            history.append((rec.config, out))
            iterations.append({"iter": i, "config": rec.config.model_dump(),
                               "status": out.status, "duration_ms": None,
                               "durations_ms": [], "samples": 0, "reason": reason})
            if i == max_iters:
                break
            continue

        dur, durations, metrics, err = _run_median(deps, episode, rec, tries)
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
                             "metrics": metrics, "rationale": rec.rationale,
                             "query_status": err["query_status"] if err else None,
                             "error": err["error"] if err else None})
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
                           "metrics": metrics,
                           "query_status": err["query_status"] if err else None,
                           "error": err["error"] if err else None})

        if no_improve >= patience or i == max_iters:
            break

    return {"best_config": best_config.model_dump() if best_config else None,
            "best_duration_ms": best_dur,
            "baseline_ms": baseline_ms,
            "baseline_metrics": baseline_metrics,
            "iterations": iterations,
            "history_len": len(history)}
