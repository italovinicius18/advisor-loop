import json, pytest
from advisor_loop.advisor.advisor import Advisor, _SUBMIT, render_history
from advisor_loop.advisor.provider import LLMResponse
from advisor_loop.episode import ConfigVector, Episode, Outcome, QueryResult
from advisor_loop.journal import Journal
from advisor_loop.probes.budget import ProbeBudgetExceeded
from advisor_loop.probes.duckdb_probe import ProbeQueryError

CFG = {"executor_cores": 4, "executor_memory_gb": 14, "executor_instances": 4,
       "driver_memory_gb": 8, "shuffle_partitions": 400, "advisory_partition_mb": 64,
       "aqe_enabled": True, "broadcast_threshold_mb": 10}

class FakeClient:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.last_kwargs = None
    def chat(self, model, messages, tools=None, temperature=0.2, max_tokens=None,
             tool_choice=None, reasoning=None):
        self.last_kwargs = {"model": model, "messages": messages, "tools": tools,
                             "temperature": temperature, "max_tokens": max_tokens,
                             "tool_choice": tool_choice, "reasoning": reasoning}
        return self.scripted.pop(0)

class FakeProbe:
    """Probe fake que sempre levanta ProbeBudgetExceeded em table_stats."""
    def table_stats(self, table):
        raise ProbeBudgetExceeded("budget esgotado")
    def available_tables(self):
        return ["store_sales", "store_returns", "date_dim"]
    def probe_budget(self):
        return 8


class FakeProbeNoBudgetIntrospection:
    """Probe fake sem probe_budget() - simula um probe mais simples/antigo."""
    def available_tables(self):
        return ["store_sales"]


class FakeProbeQueryError:
    """Probe fake que sempre levanta ProbeQueryError em table_stats - simula
    o caso real: sum(VARCHAR) contra Silver nao tipada."""
    def available_tables(self):
        return ["store_sales"]
    def table_stats(self, table):
        raise ProbeQueryError("No function matches 'sum(VARCHAR)'")

def _submit_call():
    args = json.dumps({**CFG, "rationale": "workload medio"})
    return LLMResponse(message={}, cost_usd=0.001, tool_calls=[
        {"id": "1", "function": {"name": "submit_recommendation", "arguments": args}}])

def _probe_call():
    return LLMResponse(message={}, cost_usd=0.0005, tool_calls=[
        {"id": "p1", "function": {"name": "table_stats",
                                   "arguments": json.dumps({"table": "store_sales"})}}])

def test_advise_returns_recommendation_and_logs(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    adv = Advisor(FakeClient([_submit_call()]), j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    rec = adv.advise(ep, context="ctx", probe=None, memories=[])
    assert rec.config.executor_cores == 4
    assert j.events("llm_call")[0]["cost_usd"] == 0.001

def test_advise_gives_up_after_max_rounds(tmp_path):
    empty = LLMResponse(message={"content": "hmm"}, tool_calls=[])
    adv = Advisor(FakeClient([empty] * 12), Journal(tmp_path / "j.jsonl"))
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    with pytest.raises(Exception):
        adv.advise(ep, context="ctx", probe=None, memories=[])

def test_advise_handles_probe_error_and_still_submits(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    adv = Advisor(FakeClient([_probe_call(), _submit_call()]), j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    rec = adv.advise(ep, context="ctx", probe=FakeProbe(), memories=[])
    assert rec.config.executor_cores == 4
    probe_events = j.events("probe")
    assert len(probe_events) == 1
    assert "budget esgotado" in probe_events[0]["error"]


def _submit_call_out_of_bounds():
    """driver_memory_gb=99 viola le=16 de ConfigVector -> pydantic.ValidationError."""
    bad = {**CFG, "driver_memory_gb": 99}
    args = json.dumps({**bad, "rationale": "config invalida propositalmente"})
    return LLMResponse(message={}, cost_usd=0.001, tool_calls=[
        {"id": "1", "function": {"name": "submit_recommendation", "arguments": args}}])


def test_submit_schema_declares_bounds_for_driver_memory_gb():
    props = _SUBMIT["function"]["parameters"]["properties"]
    assert props["driver_memory_gb"].get("minimum") == 4
    assert props["driver_memory_gb"].get("maximum") == 16


def test_advise_retries_after_out_of_bound_submit(tmp_path):
    """Um ValidationError do pydantic nao deve propagar (crash da CLI) - o advisor
    deve devolver o erro ao modelo como tool result e deixar ele tentar de novo."""
    j = Journal(tmp_path / "j.jsonl")
    adv = Advisor(FakeClient([_submit_call_out_of_bounds(), _submit_call()]), j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    rec = adv.advise(ep, context="ctx", probe=None, memories=[])
    assert rec.config.driver_memory_gb == 8
    assert rec.config.executor_cores == 4


def _unknown_tool_call():
    return LLMResponse(message={}, cost_usd=0.0002, tool_calls=[
        {"id": "u1", "function": {"name": "frobnicate", "arguments": "{}"}}])


def test_advise_injects_real_table_list_when_probe_given(tmp_path):
    """When a probe is active, the real table names must be injected into the
    user context up front (turn 1) so the model has nothing to guess -
    prevents hallucinated table names from other benchmarks (SSB/TPC-H)."""
    j = Journal(tmp_path / "j.jsonl")
    client = FakeClient([_submit_call()])
    adv = Advisor(client, j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    adv.advise(ep, context="ctx", probe=FakeProbe(), memories=[])
    user_msg = client.last_kwargs["messages"][1]["content"]
    assert "Tabelas reais disponiveis" in user_msg
    assert "store_sales" in user_msg
    assert "store_returns" in user_msg
    assert "date_dim" in user_msg
    system_msg = client.last_kwargs["messages"][0]["content"]
    assert "list_tables" in system_msg
    assert "NUNCA invente" in system_msg


def test_advise_system_prompt_forbids_table_name_invention_without_probe(tmp_path):
    """The table-discipline instruction is a static system-prompt line - it
    must be present even at L0 (no probe), it's just inert there."""
    j = Journal(tmp_path / "j.jsonl")
    client = FakeClient([_submit_call()])
    adv = Advisor(client, j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    adv.advise(ep, context="ctx", probe=None, memories=[])
    system_msg = client.last_kwargs["messages"][0]["content"]
    assert "NUNCA invente" in system_msg
    user_msg = client.last_kwargs["messages"][1]["content"]
    assert "Tabelas reais disponiveis" not in user_msg


def test_advise_injects_probe_budget_line_when_available(tmp_path):
    """The advisor must see its probe budget up front so it can prioritize
    the big fact tables instead of blowing the budget on trivial dimensions."""
    j = Journal(tmp_path / "j.jsonl")
    client = FakeClient([_submit_call()])
    adv = Advisor(client, j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    adv.advise(ep, context="ctx", probe=FakeProbe(), memories=[])
    user_msg = client.last_kwargs["messages"][1]["content"]
    assert "Orcamento de sondagem" in user_msg
    assert "8" in user_msg


def test_advise_skips_budget_line_for_probe_without_introspection(tmp_path):
    """A probe lacking probe_budget() must not crash advise() - the budget
    line is simply omitted (getattr defensively)."""
    j = Journal(tmp_path / "j.jsonl")
    client = FakeClient([_submit_call()])
    adv = Advisor(client, j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    adv.advise(ep, context="ctx", probe=FakeProbeNoBudgetIntrospection(), memories=[])
    user_msg = client.last_kwargs["messages"][1]["content"]
    assert "Orcamento de sondagem" not in user_msg


def test_advise_injects_try_cast_typing_note_when_probe_given(tmp_path):
    """Silver columns are all VARCHAR (untyped) - the advisor must be told to
    TRY_CAST before aggregating, or it will keep hitting sum(VARCHAR) errors."""
    j = Journal(tmp_path / "j.jsonl")
    client = FakeClient([_submit_call()])
    adv = Advisor(client, j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    adv.advise(ep, context="ctx", probe=FakeProbe(), memories=[])
    user_msg = client.last_kwargs["messages"][1]["content"]
    assert "TRY_CAST" in user_msg


def test_advise_handles_probe_query_error_and_still_submits(tmp_path):
    """A ProbeQueryError (e.g. sum(VARCHAR) BinderException against untyped
    Silver) must become a tool-error fed back to the model, not a campaign
    crash - the loop must continue to a valid submit_recommendation."""
    j = Journal(tmp_path / "j.jsonl")
    adv = Advisor(FakeClient([_probe_call(), _submit_call()]), j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    rec = adv.advise(ep, context="ctx", probe=FakeProbeQueryError(), memories=[])
    assert rec.config.executor_cores == 4
    probe_events = j.events("probe")
    assert len(probe_events) == 1
    assert "sum(VARCHAR)" in probe_events[0]["error"]


def test_advise_forwards_max_tokens_when_configured(tmp_path):
    """Advisor(max_tokens=...) must be forwarded on every client.chat() call."""
    j = Journal(tmp_path / "j.jsonl")
    client = FakeClient([_submit_call()])
    adv = Advisor(client, j, max_tokens=512)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    adv.advise(ep, context="ctx", probe=None, memories=[])
    assert client.last_kwargs["max_tokens"] == 512


def test_advise_max_tokens_defaults_to_none(tmp_path):
    """Default Advisor (no max_tokens) must not cap output - preserves old behavior."""
    j = Journal(tmp_path / "j.jsonl")
    client = FakeClient([_submit_call()])
    adv = Advisor(client, j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    adv.advise(ep, context="ctx", probe=None, memories=[])
    assert client.last_kwargs["max_tokens"] is None


def test_advise_handles_hallucinated_tool_name_and_still_submits(tmp_path):
    """A hallucinated/unknown tool name must not crash the campaign (AttributeError) -
    it should become a structured tool-error message fed back to the model, and the
    loop continues to a valid submit_recommendation."""
    j = Journal(tmp_path / "j.jsonl")
    adv = Advisor(FakeClient([_unknown_tool_call(), _submit_call()]), j)
    ep = Episode(workload="w", dataset="sf100", queries=["q1"], provider="m")
    rec = adv.advise(ep, context="ctx", probe=None, memories=[])
    assert rec.config.executor_cores == 4
    probe_events = j.events("probe")
    assert len(probe_events) == 1
    assert probe_events[0]["tool"] == "frobnicate"
    assert "error" in probe_events[0]


def test_render_history_lines():
    cv = ConfigVector(**CFG)
    hist = [
        (cv, Outcome(status="completed",
                      per_query={"q": QueryResult(status="success", duration_ms=4300.0)})),
        (ConfigVector(**{**CFG, "executor_instances": 12}), Outcome(status="oom")),
    ]
    s = render_history(hist)
    assert "4300" in s and "oom" in s and "12w" in s


def test_render_history_includes_metrics():
    from advisor_loop.advisor.advisor import fmt_metrics
    cv = ConfigVector(**CFG)
    m = {"input_bytes": 1_500_000, "shuffle_write_bytes": 2 * 1024**3,
         "memory_spill_bytes": 1024**2, "gc_time_ms": 3400}
    hist = [(cv, Outcome(status="completed",
                         per_query={"q": QueryResult(status="success",
                                                     duration_ms=19507.1, metrics=m)}))]
    s = render_history(hist)
    assert "19507" in s
    assert "shuffle_w=2.0GB" in s and "spill=1.0MB" in s and "gc=3400ms" in s
    # métricas ausentes/zero não poluem a linha
    assert fmt_metrics({}) == "" and fmt_metrics({"input_bytes": 0}) == ""


def test_render_history_sorts_best_last_and_shows_total_cores():
    """OPRO-style: sucessos ordenados do pior ao melhor tempo (melhor por ultimo) e
    cada linha expoe os cores totais (paralelismo), sinal que o LLM nao calibra."""
    slow = ConfigVector(**{**CFG, "executor_cores": 2, "executor_instances": 2})   # 4 cores
    fast = ConfigVector(**{**CFG, "executor_cores": 8, "executor_instances": 8})   # 64 cores
    hist = [
        (fast, Outcome(status="completed", per_query={"q": QueryResult(
            status="success", duration_ms=1000.0)})),
        (slow, Outcome(status="completed", per_query={"q": QueryResult(
            status="success", duration_ms=5000.0)})),
    ]
    s = render_history(hist)
    assert "cores_tot=64" in s and "cores_tot=4" in s
    # a melhor (1000ms) vem DEPOIS da pior (5000ms) no texto
    assert s.index("5000ms") < s.index("1000ms")


def test_derived_signals_gc_fraction_and_records():
    from advisor_loop.advisor.advisor import _derived_signals
    cv = ConfigVector(**CFG)  # 4c x 4w = 16 cores totais, 14/4 = 3.5g/core
    qr = QueryResult(status="success", duration_ms=10000.0,
                     metrics={"gc_time_ms": 2500, "records_read": 288_000_000})
    d = _derived_signals(cv, qr)
    assert "cores_tot=16" in d and "mem/core=3.5g" in d
    assert "gc/wall=25%" in d and "rec=288M" in d


def test_history_injected_into_context(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    client = FakeClient([_submit_call()])
    adv = Advisor(client, j)
    ep = Episode(workload="w", dataset="sf1", queries=["q03"], provider="m")
    cv = ConfigVector(**CFG)
    adv.advise(ep, "ctx", None, [], history=[
        (cv, Outcome(status="completed",
                      per_query={"q03": QueryResult(status="success", duration_ms=4300.0)})),
    ])
    user_msg = client.last_kwargs["messages"][1]["content"]
    assert "4300" in user_msg


def _hist_scaleout_thin():
    """Historico que reproduz o modo de falha do smoke: so executores finos e
    numerosos (>4 instancias, <48g cada), com GC alto na melhor config."""
    from advisor_loop.episode import Outcome, QueryResult
    def cfg(inst, mem):
        return ConfigVector(**{**CFG, "executor_instances": inst, "executor_memory_gb": mem})
    return [
        (cfg(10, 30), Outcome(status="completed", per_query={"q": QueryResult(
            status="success", duration_ms=200000.0,
            metrics={"gc_time_ms": 120000, "shuffle_write_bytes": 9 * 1024**3})})),
        (cfg(12, 32), Outcome(status="completed", per_query={"q": QueryResult(
            status="success", duration_ms=183000.0,
            metrics={"gc_time_ms": 90000, "shuffle_write_bytes": 9 * 1024**3})})),
    ]


def test_history_hints_flags_untried_few_fat_and_high_gc():
    from advisor_loop.advisor.advisor import _history_hints
    h = _history_hints(_hist_scaleout_thin()).lower()
    # nudge de cobertura: reporta o que ja testou e aponta a direcao nao explorada
    assert "poucos executores" in h and "memoria alta" in h
    # dica de GC: diagnostica concorrencia-por-heap (reduzir cores/exec), NAO
    # "menos executores" monotono
    assert "gc domina" in h and "executor_cores" in h
    assert "menos executores" not in h
    # shuffle-aware: a melhor config era shuffle-heavy (9GB) -> preserva paralelismo
    assert "shuffle" in h and ("cores totais" in h or "paralelismo" in h)


def test_history_hints_local_search_and_bracketing():
    """Com uma melhor config conhecida, o nudge pede busca local + bracketing dos
    cores totais (calibracao de paralelismo), sem empurrar 'menos executores'."""
    from advisor_loop.advisor.advisor import _history_hints
    h = _history_hints(_hist_scaleout_thin()).lower()
    assert "busca local" in h
    assert "bracketing" in h and "cores totais" in h
    assert "menos executores" not in h


def test_history_hints_cores_gap_and_no_shuffle_clause_when_light():
    """Modo de falha do sonnet/deepseek: cores/exec saturados no cap (24) com
    pouco shuffle. O nudge deve (a) apontar cores/exec MODERADOS como regiao nao
    explorada e (b) NAO pedir para preservar cores totais (query nao e shuffle-
    heavy), evitando o conselho que sub-paralelizou a q50."""
    from advisor_loop.advisor.advisor import _history_hints
    from advisor_loop.episode import Outcome, QueryResult
    cv = ConfigVector(**{**CFG, "executor_cores": 24, "executor_instances": 4,
                         "executor_memory_gb": 90})
    h = _history_hints([(cv, Outcome(status="completed", per_query={"q": QueryResult(
        status="success", duration_ms=200000.0,
        metrics={"gc_time_ms": 120000, "shuffle_write_bytes": 10 * 1024 ** 2})})) ]).lower()
    assert "cores/exec moderados" in h
    assert "gc domina" in h and "executor_cores" in h
    # shuffle leve (10MB) -> sem a clausula de preservar cores totais
    assert "pesada em shuffle" not in h


def test_history_hints_no_coverage_gap_when_space_covered():
    """Se o historico ja cobre poucos executores E memoria alta, nao insiste no
    nudge de cobertura (evita ruido)."""
    from advisor_loop.advisor.advisor import _history_hints
    from advisor_loop.episode import Outcome, QueryResult
    cv = ConfigVector(**{**CFG, "executor_instances": 4, "executor_memory_gb": 48})
    h = _history_hints([(cv, Outcome(status="completed", per_query={"q": QueryResult(
        status="success", duration_ms=50000.0, metrics={})}))]).lower()
    assert "ainda nao explorou" not in h


def test_hints_injected_into_context(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    client = FakeClient([_submit_call()])
    Advisor(client, j).advise(
        Episode(workload="w", dataset="sf100", queries=["q23"], provider="m"),
        "ctx", None, [], history=_hist_scaleout_thin())
    user_msg = client.last_kwargs["messages"][1]["content"]
    assert "DICAS:" in user_msg and "poucos executores" in user_msg.lower()
