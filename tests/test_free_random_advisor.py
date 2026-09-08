from advisor_loop.studies.advisors import FreeRandomAdvisor
from advisor_loop.episode import ConfigVector, Episode
from advisor_loop.guardian import Guardian

CAPS = {"max_cores_per_container": 24, "max_memory_gb_per_container": 96,
        "max_pods": 40, "total_cores": 120, "total_memory_gb": 400}


def _ep():
    return Episode(workload="w", dataset="sf100", queries=["q23"], provider="m")


def _ranges_ok(c: ConfigVector):
    return (1 <= c.executor_cores <= 24 and 4 <= c.executor_memory_gb <= 96
            and 1 <= c.executor_instances <= 20 and 4 <= c.driver_memory_gb <= 16
            and 100 <= c.shuffle_partitions <= 4000 and 16 <= c.advisory_partition_mb <= 256
            and -1 <= c.broadcast_threshold_mb <= 512)


def test_all_proposals_feasible_and_in_range():
    g = Guardian(CAPS)
    adv = FreeRandomAdvisor(g, seed=1)
    hist = []
    for _ in range(8):
        rec = adv.advise(_ep(), "ctx", None, [], history=hist)
        assert _ranges_ok(rec.config)
        assert g.validate(rec.config)[0]
        hist.append((rec.config, None))


def test_deterministic():
    g = Guardian(CAPS)
    a, b = FreeRandomAdvisor(g, seed=7), FreeRandomAdvisor(g, seed=7)
    r1 = a.advise(_ep(), "c", None, [], history=[]).config
    r2 = b.advise(_ep(), "c", None, [], history=[]).config
    assert r1.config_hash() == r2.config_hash()


def test_no_repeat_within_episode():
    g = Guardian(CAPS)
    adv = FreeRandomAdvisor(g, seed=3)
    seen, hist = set(), []
    for _ in range(8):
        rec = adv.advise(_ep(), "c", None, [], history=hist)
        h = rec.config.config_hash()
        assert h not in seen
        seen.add(h)
        hist.append((rec.config, None))
