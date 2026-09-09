from advisor_loop.knowledge.pack import KnowledgePack
from advisor_loop.episode import Episode


def test_free_range_pack():
    pack = KnowledgePack.load("packs")
    txt = pack.render(Episode(workload="w", dataset="sf100", queries=["q23"], provider="m"))
    # faixa dos knobs aparece (min/max), nao um menu de perfis
    assert "executor_cores" in txt and "24" in txt
    low = txt.lower()
    assert "escolha um par" not in low
    assert "destes perfis" not in low
    assert "desta lista" not in low
    # instrucao de liberdade presente
    assert "livre" in low or "livremente" in low
    # caps do Guardian: um unico doc define cluster, com os caps do RNP
    c = pack.cluster
    assert c["max_pods"] >= 21 and c["total_cores"] >= 120


def test_feasibility_block_states_arithmetic_and_coverage():
    """O envelope explicito de viabilidade aparece com a ARITMETICA que o LLM
    precisa checar (produto cores/memoria) -- o dump YAML cru dos caps nao bastava
    (o smoke propos memoria total acima do cap). E traz o nudge de cobertura
    (muitos pequenos <-> poucos grandes) que ataca o vies scale-out."""
    pack = KnowledgePack.load("packs")
    txt = pack.render(Episode(workload="w", dataset="sf100", queries=["q23"], provider="m"))
    low = txt.lower()
    assert "restricoes rigidas" in low
    assert "executor_cores * executor_instances" in txt
    assert "executor_memory_gb * executor_instances" in txt
    # nudge de cobertura: os dois extremos do formato de executor
    assert "muitos executores" in low and "poucos executores" in low
