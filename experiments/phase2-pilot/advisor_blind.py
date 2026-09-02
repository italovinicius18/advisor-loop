"""Blind free-proposal advisor for the Phase-2 pilot (throwaway spike).

Differs from the Phase-1 study advisor in two ways, both being piloted here:
- NO discrete config menu: the LLM proposes any config within the schema ranges.
- Enriched context (T1+T2+T3): knob-cards + cluster caps, the query's distilled
  physical plan, and the full run history with real telemetry -- not just wall time.

Reuses advisor_loop's OpenRouterClient and ConfigVector (via PYTHONPATH).
"""
import json

from advisor_loop.advisor.provider import OpenRouterClient
from advisor_loop.episode import ConfigVector

from knobs import KNOB_CARDS, CLUSTER_CAPS

_SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_config",
        "description": "Propose the next Spark config to try for this query.",
        "parameters": {
            "type": "object",
            "properties": {
                "executor_cores": {"type": "integer", "minimum": 1, "maximum": 24},
                "executor_memory_gb": {"type": "integer", "minimum": 4, "maximum": 96},
                "executor_instances": {"type": "integer", "minimum": 1, "maximum": 20},
                "driver_memory_gb": {"type": "integer", "minimum": 4, "maximum": 16},
                "shuffle_partitions": {"type": "integer", "minimum": 100, "maximum": 4000},
                "advisory_partition_mb": {"type": "integer", "minimum": 16, "maximum": 256},
                "aqe_enabled": {"type": "boolean"},
                "broadcast_threshold_mb": {"type": "integer", "minimum": -1, "maximum": 512},
                "rationale": {"type": "string",
                              "description": "One sentence: why this config, given the plan/telemetry."},
            },
            "required": ["executor_cores", "executor_memory_gb", "executor_instances",
                         "driver_memory_gb", "shuffle_partitions", "advisory_partition_mb",
                         "aqe_enabled", "broadcast_threshold_mb", "rationale"],
        },
    },
}

_SYSTEM = """You are a Spark performance tuner. Goal: minimise the wall-clock \
runtime of ONE query on a fixed dataset by choosing its Spark config. This is \
BLIND online tuning: there is no menu and no known optimum -- you propose a \
config, it runs for real, and you see its runtime + telemetry + physical plan, \
then propose the next one. Use the feedback to improve each round. Prefer \
configs that are feasible on the node below; an infeasible config wastes a run.

Cluster (hard limits):
{caps}

What each knob does:
{cards}
"""


def _system_prompt():
    caps = json.dumps(CLUSTER_CAPS, indent=2)
    cards = "\n".join(f"- {k}: {v}" for k, v in KNOB_CARDS.items())
    return _SYSTEM.format(caps=caps, cards=cards)


def _history_block(history):
    """Render prior runs with telemetry (T3): config -> runtime + key metrics."""
    if not history:
        return "No runs yet. Propose a sensible first config."
    lines = []
    for i, h in enumerate(history, 1):
        cfg = h["config"]
        short = {k: cfg[k] for k in ("executor_cores", "driver_memory_gb",
                 "shuffle_partitions", "advisory_partition_mb", "aqe_enabled",
                 "broadcast_threshold_mb")}
        if h["status"] == "success":
            m = h.get("metrics", {})
            tele = (f"shuffle_read={m.get('shuffle_read_bytes',0)/1e6:.0f}MB "
                    f"spill={m.get('memory_spilled_bytes',0)/1e6:.0f}MB "
                    f"gc={m.get('gc_time_ms',0)}ms stages={m.get('num_stages','?')} "
                    f"bcast_joins={m.get('num_broadcast_joins','?')}")
            lines.append(f"run {i}: {short} -> {h['duration_ms']:.0f} ms | {tele}")
        else:
            lines.append(f"run {i}: {short} -> {h['status'].upper()} "
                         f"({h.get('error','')[:120]})")
    return "\n".join(lines)


def advise(client: OpenRouterClient, model: str, query_name: str, query_sql: str,
           default_plan: str, history: list) -> tuple[ConfigVector, str, float]:
    """Return (config, rationale, cost_usd) for the next config to try."""
    user = (f"Query: {query_name}\n\nSQL:\n{query_sql.strip()}\n\n"
            f"Physical plan under the DEFAULT config (T2):\n{default_plan}\n\n"
            f"Run history so far (most recent last):\n{_history_block(history)}\n\n"
            "Call submit_config with your next config to try.")
    resp = client.chat(
        model=model,
        messages=[{"role": "system", "content": _system_prompt()},
                  {"role": "user", "content": user}],
        tools=[_SUBMIT_TOOL],
        tool_choice="required",
        reasoning={"enabled": False},
        temperature=0.3,
    )
    args = json.loads(resp.tool_calls[0]["function"]["arguments"])
    rationale = args.pop("rationale", "")
    cfg = ConfigVector(**args)
    return cfg, rationale, resp.cost_usd
