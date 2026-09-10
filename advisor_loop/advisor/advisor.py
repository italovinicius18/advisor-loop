import json
import pydantic
from advisor_loop.episode import ConfigVector, Episode, Outcome, Recommendation
from advisor_loop.journal import Journal
from advisor_loop.probes.budget import ProbeBudgetExceeded
from advisor_loop.probes.duckdb_probe import ProbeTimeout, ProbeQueryError

class AdvisorError(Exception):
    pass

def _field_schema(name: str, field) -> dict:
    if name == "aqe_enabled":
        return {"type": "boolean"}
    schema: dict = {"type": "integer"}
    for meta in field.metadata:
        if hasattr(meta, "ge"):
            schema["minimum"] = meta.ge
        if hasattr(meta, "le"):
            schema["maximum"] = meta.le
    return schema

def _submit_tool(fields: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "submit_recommendation",
            "description": "Envia a configuracao final recomendada.",
            "parameters": {"type": "object", "additionalProperties": False,
                           "required": [*fields, "rationale"],
                           "properties": {**{k: _field_schema(k, ConfigVector.model_fields[k])
                                             for k in fields},
                                          "rationale": {"type": "string"}}}}}

_SUBMIT = _submit_tool(list(ConfigVector.model_fields))

# Ablacao "matrix3": o LLM so move os 3 eixos que a grade offline varreu
# (cores x memoria x instancias); os outros 5 ficam PINADOS nos valores fixos da
# matriz (Spark default; AQE on). Motivacao (journals r5/r6): com schema livre o
# modelo espalha shuffle_partitions em 100..2000 e advisory em 16..128 -- uma
# config de forma quase igual ao oraculo (12c x 7w x 56g) ficou 2.4x mais lenta
# com sp=400/adv=128 vs sp=200/adv=64 da grade. Sem pinar, nao da para separar
# "erro de forma" de "ruido dos knobs extras" no regret.
SHAPE_KNOBS = ["executor_cores", "executor_memory_gb", "executor_instances"]
PINNED_MATRIX = {"driver_memory_gb": 8, "shuffle_partitions": 200,
                 "advisory_partition_mb": 64, "aqe_enabled": True,
                 "broadcast_threshold_mb": 10}
_SUBMIT_MATRIX3 = _submit_tool(SHAPE_KNOBS)

_KNOBS_NOTE = {
    "matrix3": ("## Knobs ajustaveis NESTA campanha\nSomente executor_cores, "
                "executor_memory_gb e executor_instances. Os demais estao FIXOS em "
                + ", ".join(f"{k}={v}" for k, v in PINNED_MATRIX.items())
                + " (nao os envie)."),
}

_PROBE_TOOLS = [
    {"type": "function", "function": {"name": "list_tables",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "table_stats",
        "parameters": {"type": "object", "required": ["table"],
                       "properties": {"table": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "sample_query",
        "parameters": {"type": "object", "required": ["sql"],
                       "properties": {"sql": {"type": "string"}}}}},
]

def _fmt_bytes(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def _fmt_count(n) -> str:
    n = float(n or 0)
    for unit in ("", "K", "M", "B"):
        if n < 1000 or unit == "B":
            return f"{n:.0f}{unit}"
        n /= 1000


def fmt_metrics(m: dict) -> str:
    """Resumo compacto das métricas reais (só campos não-zero) para o histórico
    do LLM - dá sinais causais (shuffle/spill/gc) além do tempo puro."""
    if not m:
        return ""
    parts = []
    if m.get("input_bytes"):
        parts.append(f"in={_fmt_bytes(m['input_bytes'])}")
    if m.get("shuffle_read_bytes"):
        parts.append(f"shuffle_r={_fmt_bytes(m['shuffle_read_bytes'])}")
    if m.get("shuffle_write_bytes"):
        parts.append(f"shuffle_w={_fmt_bytes(m['shuffle_write_bytes'])}")
    spill = (m.get("memory_spill_bytes", 0) or 0) + (m.get("disk_spill_bytes", 0) or 0)
    if spill:
        parts.append(f"spill={_fmt_bytes(spill)}")
    if m.get("gc_time_ms"):
        parts.append(f"gc={m['gc_time_ms']:g}ms")
    return f" ({', '.join(parts)})" if parts else ""


def _history_hints(history: list[tuple[ConfigVector, Outcome]]) -> str:
    """Dicas causais derivadas do historico desta query. O smoke k=1 mostrou
    DOIS modos de falha opostos: deepseek saturou no extremo few-fat (24c no cap,
    heaps enormes) e sonnet sub-paralelizou uma query shuffle-heavy (q50). Logo o
    nudge NAO e monotono "menos executores + mais memoria": (1) se a MELHOR config
    teve GC alto, diagnostica concorrencia-por-heap (reduzir executor_cores, nao o
    numero de executores) e, em query shuffle-heavy, pede para PRESERVAR os cores
    totais; (2) spill -> mais memoria/particoes; (3) COBERTURA: lista o que ja foi
    testado (instancias, cores/exec, memoria) e aponta direcoes moderadas ainda
    nao exploradas, sem empurrar para o extremo."""
    succ = [(cv, qr) for cv, o in history
            for qr in [next((q for q in o.per_query.values()
                             if q.status == "success" and q.duration_ms is not None), None)]
            if qr is not None]
    hints: list[str] = []
    if succ:
        best_cv, best_qr = min(succ, key=lambda t: t[1].duration_ms)
        m = best_qr.metrics or {}
        gc = m.get("gc_time_ms", 0) or 0
        dur = best_qr.duration_ms or 0
        shuffle = max(m.get("shuffle_read_bytes", 0) or 0,
                      m.get("shuffle_write_bytes", 0) or 0)
        heavy_shuffle = shuffle >= 1 * 1024 ** 3  # >=1GB => limitada por paralelismo
        if gc and dur and gc >= 0.25 * dur:
            tip = (f"GC domina a melhor config ate agora (gc={gc:g}ms vs wall={dur:g}ms, "
                   f"{best_cv.executor_cores}c x {best_cv.executor_instances}w x "
                   f"{best_cv.executor_memory_gb}g): GC vem tipicamente de tarefas "
                   "concorrentes demais por heap. Reduza executor_cores por executor e/ou "
                   "aumente executor_memory_gb (nao sature executor_cores no maximo).")
            if heavy_shuffle:
                tip += (f" Esta query e pesada em shuffle ({_fmt_bytes(shuffle)}): NAO "
                        "derrube os cores totais (paralelismo) -- se cortar cores por "
                        "executor, compense com mais executores e/ou aumente shuffle_partitions.")
            hints.append(tip)
        spill = (m.get("memory_spill_bytes", 0) or 0) + (m.get("disk_spill_bytes", 0) or 0)
        if spill:
            hints.append("Houve spill: aumente memoria por executor ou shuffle_partitions.")
        u = utilization(best_cv, dur, m)
        if u is not None and u < 0.3:
            hints.append(
                f"UTILIZACAO BAIXA na melhor config ({100 * u:.0f}% dos "
                f"{best_cv.executor_cores * best_cv.executor_instances} cores totais ocupados): "
                "a query nao e limitada por CPU -- cortar cores totais nao deve piorar e "
                "reduz overhead de scheduling/inicializacao. Teste MENOS cores totais.")
        elif u is not None and u > 0.7:
            hints.append(
                f"UTILIZACAO ALTA na melhor config ({100 * u:.0f}%): limitada por CPU -- "
                "mais cores totais devem reduzir o tempo. Teste MAIS cores totais.")
        # Trust-region + bracketing (TuRBO/OPRO): com uma regiao boa ja conhecida,
        # gastar orcamento refinando LOCALMENTE em torno dela bate saltos aleatorios;
        # e o bracketing dos cores totais ataca direto a falha de calibracao de
        # paralelismo (achar o joelho da curva tempo x cores totais).
        tc = best_cv.executor_cores * best_cv.executor_instances
        hints.append(
            f"MELHOR ate agora: {best_cv.executor_cores}c x {best_cv.executor_instances}w "
            f"= {tc} cores totais ({best_qr.duration_ms:g}ms). Faca BUSCA LOCAL em torno dela: "
            "mude UM eixo por vez em passos pequenos (~25-50%), nao salte para configs distantes. "
            f"Para calibrar o paralelismo, faca BRACKETING dos cores totais: proponha uma config "
            f"com MAIS que {tc} cores totais e, noutra iteracao, uma com MENOS cores totais, para "
            "achar o joelho da curva tempo x cores totais.")
    if history:
        inst = sorted({cv.executor_instances for cv, _ in history})
        mem = sorted({cv.executor_memory_gb for cv, _ in history})
        cores = sorted({cv.executor_cores for cv, _ in history})
        cov = (f"Ja testou executor_instances={inst}, cores/exec={cores} e "
               f"memory_gb/exec={mem}.")
        gaps = []
        if inst and min(inst) > 4:
            gaps.append("poucos executores (ex.: 4-8)")
        if mem and max(mem) < 48:
            gaps.append("memoria alta por executor (ex.: 48-64g)")
        if cores and min(cores) >= 24:
            gaps.append("cores/exec moderados (ex.: 8-12) mantendo os cores totais")
        if gaps:
            cov += " Ainda NAO explorou: " + " e ".join(gaps) + " -- experimente na proxima."
        hints.append(cov)
    return " ".join(hints)


def _derived_signals(cv: ConfigVector, qr) -> str:
    """Sinais derivados que o LLM sozinho erra ao raciocinar em prosa: paralelismo
    total (cores_tot = cores/exec * instancias), memoria por tarefa (mem/core) e a
    fracao de GC sobre o wall. Expor esses numeros ao lado do tempo torna EXPLICITA
    a relacao paralelismo<->tempo que o modelo nao calibra (super-provisiona query
    leve, sub-provisiona query pesada)."""
    tc = cv.executor_cores * cv.executor_instances
    mpc = cv.executor_memory_gb / cv.executor_cores
    parts = [f"cores_tot={tc}", f"mem/core={mpc:.1f}g"]
    m = qr.metrics or {}
    gc = m.get("gc_time_ms") or 0
    if gc and qr.duration_ms:
        parts.append(f"gc/wall={100 * gc / qr.duration_ms:.0f}%")
    u = utilization(cv, qr.duration_ms, m)
    if u is not None:
        parts.append(f"util={100 * u:.0f}%")
    if m.get("num_tasks"):
        parts.append(f"tasks={_fmt_count(m['num_tasks'])}")
    if m.get("max_task_ms") and m.get("median_task_ms"):
        parts.append(f"skew={m['max_task_ms'] / m['median_task_ms']:.1f}x")
    if m.get("records_read"):
        parts.append(f"rec={_fmt_count(m['records_read'])}")
    return ", ".join(parts)


def utilization(cv: ConfigVector, duration_ms, m: dict) -> float | None:
    """Fracao dos cores totais realmente ocupados com CPU durante o wall:
    executor_cpu_time / (cores_tot x wall). Baixa (<30%) => a query NAO e limitada
    por CPU (I/O, latencia, overhead de scheduling) e mais cores nao ajudam."""
    cpu = (m or {}).get("executor_cpu_time_ms") or 0
    tc = cv.executor_cores * cv.executor_instances
    if not cpu or not duration_ms or not tc:
        return None
    return cpu / (tc * duration_ms)


def render_profile(cv: ConfigVector, duration_ms, m: dict) -> str:
    """Perfil da query medido na ITERACAO 0 (config default): trabalho total em
    core-segundos, utilizacao, volume de I/O/shuffle e forma do plano fisico.
    Da ao LLM o PESO real da query antes da primeira proposta -- e o insumo que
    faltava para calibrar paralelismo ao peso (super-provisionava a leve,
    sub-provisionava a pesada). Vem da execucao propria, nunca da matriz."""
    m = m or {}
    tc = cv.executor_cores * cv.executor_instances
    lines = [f"## Perfil da query (iteracao 0, config default "
             f"{cv.executor_cores}c x {cv.executor_instances}w x {cv.executor_memory_gb}g "
             f"= {tc} cores totais)"]
    if duration_ms:
        lines.append(f"- tempo: {duration_ms / 1000:.1f}s")
    cpu = m.get("executor_cpu_time_ms") or 0
    if cpu:
        work = cpu / 1000
        lines.append(f"- trabalho de CPU total: {work:.0f} core-segundos "
                     f"(limite inferior ideal com N cores totais = {work:.0f}/N s)")
        u = utilization(cv, duration_ms, m)
        if u is not None:
            lines.append(f"- utilizacao dos cores no baseline: {100 * u:.0f}%")
    io = []
    if m.get("input_bytes"):
        io.append(f"input={_fmt_bytes(m['input_bytes'])}")
    if m.get("records_read"):
        io.append(f"registros={_fmt_count(m['records_read'])}")
    sh = max(m.get("shuffle_read_bytes") or 0, m.get("shuffle_write_bytes") or 0)
    if sh:
        io.append(f"shuffle={_fmt_bytes(sh)}")
    if io:
        lines.append("- volume: " + ", ".join(io))
    if m.get("num_tasks"):
        t = f"- tasks: {_fmt_count(m['num_tasks'])} em {m.get('num_stages', 0)} stages"
        if m.get("max_task_ms") and m.get("median_task_ms"):
            t += (f" (task mediana {m['median_task_ms'] / 1000:.1f}s, "
                  f"max {m['max_task_ms'] / 1000:.1f}s)")
        lines.append(t)
    if m.get("plan_lines"):
        lines.append(f"- plano fisico: {m.get('plan_joins', 0)} joins "
                     f"({m.get('plan_broadcast_joins', 0)} broadcast, "
                     f"{m.get('plan_sort_merge_joins', 0)} sort-merge), "
                     f"{m.get('plan_exchanges', 0)} shuffles (Exchange), "
                     f"{m.get('plan_aggregates', 0)} agregacoes, "
                     f"{m.get('plan_sorts', 0)} sorts, {m.get('plan_scans', 0)} scans")
    lines.append("Dimensione o paralelismo ao PESO acima: se o trabalho de CPU e pequeno "
                 "e a utilizacao baixa, muitos cores so adicionam overhead; se o trabalho "
                 "e grande e a utilizacao alta, mais cores totais reduzem o tempo.")
    return "\n".join(lines)


def render_history(history: list[tuple[ConfigVector, Outcome]]) -> str:
    """Trajetoria realimentada ao advisor, formatada estilo OPRO (Yang et al.,
    ICLR 2024): as execucoes com SUCESSO vem ORDENADAS por tempo (pior -> melhor),
    de modo que a MELHOR config fique por ULTIMO (mais perto do ponto de geracao) e
    o modelo infira a direcao de melhora a partir do ranking, em vez de ler uma
    ordem cronologica sem sinal. Cada linha carrega config + sinais derivados
    (cores totais, mem/core, gc/wall) + metricas reais (shuffle/spill/gc). As falhas
    vem antes, agrupadas, para nao poluir o ranking dos sucessos."""
    def prefix(cv):
        return f"{cv.executor_cores}c/{cv.executor_memory_gb}g/{cv.executor_instances}w"
    succ, fail = [], []
    for cv, outcome in history:
        qr = next((q for q in outcome.per_query.values()
                   if q.status == "success" and q.duration_ms is not None), None)
        (succ if qr is not None else fail).append((cv, outcome, qr))
    lines = [f"{prefix(cv)} → {outcome.status}" for cv, outcome, _ in fail]
    succ.sort(key=lambda t: t[2].duration_ms, reverse=True)  # pior primeiro, melhor por ultimo
    for cv, _, qr in succ:
        lines.append(f"{prefix(cv)} ({_derived_signals(cv, qr)}) → "
                     f"{qr.duration_ms:g}ms{fmt_metrics(qr.metrics)}")
    if succ:
        lines.append("(acima: ordenado do PIOR ao MELHOR tempo; a ULTIMA linha e a "
                     "melhor config ate agora -- proponha uma que a supere.)")
    return "\n".join(lines)


class Advisor:
    def __init__(self, client, journal: Journal, max_rounds: int = 12,
                 max_tokens: int | None = None,
                 tool_choice: str | dict | None = None,
                 reasoning: dict | None = None,
                 knobs: str = "free"):
        self.client, self.journal, self.max_rounds = client, journal, max_rounds
        self.max_tokens = max_tokens
        # tool_choice/reasoning: passados a cada chat() -> forçam tool call e/ou
        # desligam reasoning para manter o modelo rápido e no protocolo.
        self.tool_choice = tool_choice
        self.reasoning = reasoning
        # knobs: "free" (8 knobs, schema livre) ou "matrix3" (ablacao: so os 3
        # eixos da grade; os demais pinados em PINNED_MATRIX).
        if knobs not in ("free", "matrix3"):
            raise ValueError(f"knobs desconhecido: {knobs!r} (use 'free' ou 'matrix3')")
        self.knobs = knobs
        self._submit = _SUBMIT_MATRIX3 if knobs == "matrix3" else _SUBMIT
        self._pinned = PINNED_MATRIX if knobs == "matrix3" else {}

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list[tuple[ConfigVector, Outcome]] = []) -> Recommendation:
        tools = [self._submit] + (_PROBE_TOOLS if probe else [])
        mem = "\n".join(f"- {m}" for m in memories) or "(sem memorias)"
        user_content = f"{context}\n\n## Memorias de episodios anteriores\n{mem}"
        if self.knobs in _KNOBS_NOTE:
            user_content = f"{user_content}\n\n{_KNOBS_NOTE[self.knobs]}"
        if history:
            hints = _history_hints(history)
            user_content = (
                f"{user_content}\n\n## Iteracoes anteriores nesta query (min_time)\n"
                f"{render_history(history)}\nProponha uma config que REDUZA o tempo; "
                "se alguma falhou por OOM ou foi inviavel, ajuste memoria/paralelismo "
                "e respeite as restricoes rigidas."
                + (f"\nDICAS: {hints}" if hints else ""))
        if probe is not None:
            user_content = (
                "IMPORTANTE: as colunas das tabelas Silver sao todas VARCHAR (nao "
                "tipadas). Para qualquer agregacao/aritmetica use TRY_CAST(coluna AS "
                "DOUBLE) - ex.: sum(TRY_CAST(ss_quantity AS DOUBLE)). count(*) e "
                f"count(col) funcionam sem cast.\n\n{user_content}")
            probe_budget = getattr(probe, "probe_budget", None)
            if probe_budget is not None:
                user_content = (
                    f"Orcamento de sondagem: no maximo {probe_budget()} probes "
                    "(1 chamada de list_tables/table_stats/sample_query = 1 probe). "
                    "Priorize as tabelas MAIORES e mais relevantes para o custo da query "
                    "(fatos: store_sales/catalog_sales/web_sales/inventory); nao gaste "
                    "probes em dimensoes pequenas. Se nao souber quais tabelas a query "
                    f"toca, olhe a lista real acima.\n\n{user_content}")
            user_content = (f"Tabelas reais disponiveis neste dataset: "
                             f"{', '.join(probe.available_tables())}.\n\n{user_content}")
        messages = [
            {"role": "system", "content":
             "Seja conciso: rationale de no maximo 120 palavras. Nao repita o contexto. "
             "Ao sondar dados: chame list_tables PRIMEIRO e sonde SOMENTE tabelas exatamente "
             "como retornadas por ela. NUNCA invente ou adivinhe nomes de tabela (nao use "
             "nomes de outros benchmarks como lineorder/part/supplier). Um table_stats com "
             "erro significa que a tabela nao existe - volte para a lista real. "
             "Voce recomenda configuracao de execucao para cargas SQL analiticas. "
             "Use as tools de sondagem se disponiveis e finalize SEMPRE via submit_recommendation."},
            {"role": "user", "content": user_content},
        ]
        for _ in range(self.max_rounds):
            r = self.client.chat(ep.provider, messages, tools=tools,
                                 max_tokens=self.max_tokens,
                                 tool_choice=self.tool_choice, reasoning=self.reasoning)
            self.journal.append({"kind": "llm_call", "model": ep.provider,
                                 "cost_usd": r.cost_usd, "usage": r.usage})
            if not r.tool_calls:
                messages.append({"role": "assistant", "content": r.message.get("content") or ""})
                messages.append({"role": "user", "content":
                                 "Finalize chamando submit_recommendation."})
                continue
            messages.append({"role": "assistant", "content": None, "tool_calls": r.tool_calls})
            for call in r.tool_calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                if name == "submit_recommendation":
                    rationale = args.pop("rationale", "")
                    try:
                        # pinados por ULTIMO: mesmo que o modelo envie um knob
                        # fixo (o schema o proibe, mas nao confiamos), ele e sobrescrito.
                        config = ConfigVector(**{**args, **self._pinned})
                    except pydantic.ValidationError as e:
                        self.journal.append({"kind": "probe", "tool": name, "args": args,
                                             "error": str(e)})
                        messages.append({"role": "tool", "tool_call_id": call["id"],
                                         "content": json.dumps({"error": f"config invalida: {e}"})})
                        continue
                    return Recommendation(config=config, rationale=rationale)
                try:
                    if probe is None:
                        raise AttributeError(f"tool '{name}' indisponivel: probe nao ativo neste nivel")
                    result = getattr(probe, name)(**args)
                except (ProbeBudgetExceeded, ProbeTimeout, ProbeQueryError, ValueError,
                        KeyError, AttributeError) as e:
                    self.journal.append({"kind": "probe", "tool": name, "args": args,
                                         "error": str(e)})
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps({"error": str(e)})})
                    continue
                self.journal.append({"kind": "probe", "tool": name, "args": args})
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, default=str)[:4000]})
        raise AdvisorError(f"sem submit_recommendation apos {self.max_rounds} rounds")
