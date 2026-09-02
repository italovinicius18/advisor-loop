"""Estudo de regret offline: família de advisores + runner.

Parquet sintético (2 SFs x 2 queries x 3 configs) com números escolhidos para
que oráculo, regret e a predição do surrogate sejam verificáveis à mão.

Configs: A=(4,14,4) B=(8,28,2) C=(12,48,6). Oráculos: q1->B, q2->A (nos 2 SFs).
best-on-average = A (norm média 1.125 < B 1.4 < C 1.35).
"""
import duckdb

from advisor_loop.studies.advisors import (
    config_space, best_fixed_config, FixedAdvisor, RandomAdvisor, SurrogateAdvisor,
)
from advisor_loop.studies.regret import AdvisorSpec, run_study


def _ref(tmp_path):
    p = tmp_path / "ref.parquet"
    duckdb.sql("""SELECT * FROM (VALUES
        ('sf100','q1',4,14,4,'success',1000.0),
        ('sf100','q1',8,28,2,'success', 800.0),
        ('sf100','q1',12,48,6,'success',1200.0),
        ('sf100','q2',4,14,4,'success', 500.0),
        ('sf100','q2',8,28,2,'success', 900.0),
        ('sf100','q2',12,48,6,'success', 600.0),
        ('sf250','q1',4,14,4,'success',2000.0),
        ('sf250','q1',8,28,2,'success',1600.0),
        ('sf250','q1',12,48,6,'success',2400.0),
        ('sf250','q2',4,14,4,'success',1000.0),
        ('sf250','q2',8,28,2,'success',1800.0),
        ('sf250','q2',12,48,6,'success',1200.0))
        t(dataset,query,executor_cores,executor_memory_gb,executor_instances,status,duration_ms)
    """).write_parquet(str(p))
    return str(p)


def test_config_space_lists_distinct_grid_configs(tmp_path):
    space = config_space(_ref(tmp_path))
    triples = {(c.executor_cores, c.executor_memory_gb, c.executor_instances) for c in space}
    assert triples == {(4, 14, 4), (8, 28, 2), (12, 48, 6)}


def test_best_fixed_is_lowest_normalized(tmp_path):
    cfg = best_fixed_config(_ref(tmp_path))
    assert (cfg.executor_cores, cfg.executor_memory_gb, cfg.executor_instances) == (4, 14, 4)


def test_surrogate_transfers_best_of_other_scale(tmp_path):
    ref = _ref(tmp_path)
    a2 = SurrogateAdvisor(ref)
    # alvo sf100/q1: melhor config de q1 no outro SF (sf250) é B=(8,28,2)
    cfg = a2._predict("sf100", "q1")
    assert (cfg.executor_cores, cfg.executor_memory_gb, cfg.executor_instances) == (8, 28, 2)


def test_run_study_regret_values(tmp_path):
    ref = _ref(tmp_path)
    space = config_space(ref)
    specs = [
        AdvisorSpec("A0-fixed", FixedAdvisor(best_fixed_config(ref)), max_iters=1),
        AdvisorSpec("A1-random", RandomAdvisor(space, seed=1), max_iters=3),
        AdvisorSpec("A2-surrogate", SurrogateAdvisor(ref), max_iters=1),
    ]
    res = run_study(ref, specs)

    # A0 (config A): regret médio = (0.25+0+0.25+0)/4 = 0.125
    assert abs(res["A0-fixed"]["agg"]["regret_mean"] - 0.125) < 1e-9
    assert res["A0-fixed"]["agg"]["feasibility_hit_rate"] == 1.0

    # A2 zero-shot acerta o oráculo em todos (config ótima é escala-invariante aqui)
    assert res["A2-surrogate"]["agg"]["regret_mean"] == 0.0

    # A1 com 3 tries cobre as 3 configs -> acha o oráculo -> regret 0
    assert res["A1-random"]["agg"]["regret_mean"] == 0.0
    assert res["A1-random"]["agg"]["feasibility_hit_rate"] == 1.0

    # baselines não-LLM: custo zero
    assert res["A0-fixed"]["agg"]["cost_usd"] == 0.0


def test_infeasible_config_penalizes_hit_rate(tmp_path):
    # grade só com config A em sf100/q1; FixedAdvisor(B) é inviável lá -> miss.
    p = tmp_path / "one.parquet"
    duckdb.sql("""SELECT * FROM (VALUES ('sf100','q1',4,14,4,'success',1000.0))
        t(dataset,query,executor_cores,executor_memory_gb,executor_instances,status,duration_ms)
    """).write_parquet(str(p))
    from advisor_loop.studies.advisors import make_config
    specs = [AdvisorSpec("A0-B", FixedAdvisor(make_config(8, 28, 2)), max_iters=1)]
    res = run_study(str(p), specs)
    assert res["A0-B"]["agg"]["feasibility_hit_rate"] == 0.0
    assert res["A0-B"]["agg"]["regret_mean"] is None
