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

def render_history(history: list[tuple[ConfigVector, Outcome]]) -> str:
    """Uma linha por iteracao anterior desta query: config + duracao (sucesso)
    ou status (falha, ex.: oom) - realimenta o advisor com resultados reais em
    vez dele adivinhar as configs seguintes as cegas."""
    lines = []
    for i, (cv, outcome) in enumerate(history, start=1):
        prefix = (f"iter {i}: {cv.executor_cores}c/{cv.executor_memory_gb}g/"
                  f"{cv.executor_instances}w")
        dur = next((qr.duration_ms for qr in outcome.per_query.values()
                    if qr.status == "success" and qr.duration_ms is not None), None)
        if dur is not None:
            lines.append(f"{prefix} → {dur:g}ms")
        else:
            lines.append(f"{prefix} → {outcome.status}")
    return "\n".join(lines)


class Advisor:
    def __init__(self, client, journal: Journal, max_rounds: int = 12,
                 max_tokens: int | None = None):
        self.client, self.journal, self.max_rounds = client, journal, max_rounds
        self.max_tokens = max_tokens

    def advise(self, ep: Episode, context: str, probe, memories: list[str],
               history: list[tuple[ConfigVector, Outcome]] = []) -> Recommendation:
        tools = [_SUBMIT] + (_PROBE_TOOLS if probe else [])
        mem = "\n".join(f"- {m}" for m in memories) or "(sem memorias)"
        user_content = f"{context}\n\n## Memorias de episodios anteriores\n{mem}"
        if history:
            user_content = (
                f"{user_content}\n\n## Iteracoes anteriores nesta query (min_time)\n"
                f"{render_history(history)}\nProponha uma config que REDUZA o tempo; "
                "se alguma falhou por OOM, ajuste memoria/paralelismo.")
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
            r = self.client.chat(ep.provider, messages, tools=tools, max_tokens=self.max_tokens)
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
