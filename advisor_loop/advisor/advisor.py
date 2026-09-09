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

_SUBMIT = {
    "type": "function",
    "function": {
        "name": "submit_recommendation",
        "description": "Envia a configuracao final recomendada.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": [*ConfigVector.model_fields, "rationale"],
                       "properties": {**{k: _field_schema(k, f)
                                         for k, f in ConfigVector.model_fields.items()},
                                      "rationale": {"type": "string"}}}}}

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


def render_history(history: list[tuple[ConfigVector, Outcome]]) -> str:
    """Uma linha por iteracao anterior desta query: config + duracao + metricas
    reais (sucesso) ou status (falha, ex.: oom) - realimenta o advisor com
    resultados reais (tempo E shuffle/spill/gc) em vez de adivinhar as cegas."""
    lines = []
    for i, (cv, outcome) in enumerate(history, start=1):
        prefix = (f"iter {i}: {cv.executor_cores}c/{cv.executor_memory_gb}g/"
                  f"{cv.executor_instances}w")
        qr = next((q for q in outcome.per_query.values()
                   if q.status == "success" and q.duration_ms is not None), None)
        if qr is not None:
            lines.append(f"{prefix} → {qr.duration_ms:g}ms{fmt_metrics(qr.metrics)}")
        else:
            lines.append(f"{prefix} → {outcome.status}")
    return "\n".join(lines)


class Advisor:
    def __init__(self, client, journal: Journal, max_rounds: int = 12,
                 max_tokens: int | None = None,
                 tool_choice: str | dict | None = None,
                 reasoning: dict | None = None):
        self.client, self.journal, self.max_rounds = client, journal, max_rounds
        self.max_tokens = max_tokens
        # tool_choice/reasoning: passados a cada chat() -> forçam tool call e/ou
        # desligam reasoning para manter o modelo rápido e no protocolo.
        self.tool_choice = tool_choice
        self.reasoning = reasoning

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list[tuple[ConfigVector, Outcome]] = []) -> Recommendation:
        tools = [_SUBMIT] + (_PROBE_TOOLS if probe else [])
        mem = "\n".join(f"- {m}" for m in memories) or "(sem memorias)"
        user_content = f"{context}\n\n## Memorias de episodios anteriores\n{mem}"
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
                        config = ConfigVector(**args)
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
