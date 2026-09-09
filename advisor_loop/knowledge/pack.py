import hashlib
from pathlib import Path
import yaml
from advisor_loop.episode import Episode

class KnowledgePack:
    def __init__(self, docs: dict[str, dict], hash_: str):
        self._docs = docs
        self.hash = hash_

    @classmethod
    def load(cls, pack_dir: str | Path) -> "KnowledgePack":
        pack_dir = Path(pack_dir)
        blobs, docs = [], {}
        for f in sorted(pack_dir.glob("*.yaml")):
            text = f.read_text()
            blobs.append(text)
            docs[f.stem] = yaml.safe_load(text)
        h = hashlib.sha256("".join(blobs).encode()).hexdigest()[:16]
        return cls(docs, h)

    @property
    def cluster(self) -> dict:
        for doc in self._docs.values():
            if "cluster" in doc:
                return doc["cluster"]
        raise KeyError("nenhum YAML do pack define 'cluster'")

    def render(self, episode: Episode) -> str:
        parts = [f"## Episodio\nworkload={episode.workload} dataset={episode.dataset} "
                 f"queries={','.join(episode.queries)} objetivo={episode.objective}"]
        for name, doc in self._docs.items():
            parts.append(f"## {name}\n{yaml.safe_dump(doc, allow_unicode=True)}")
        try:
            parts.append(self._feasibility_block())
        except KeyError:
            pass  # pack sem doc 'cluster' (testes minimos) -> sem envelope explicito
        return "\n\n".join(parts)

    def _feasibility_block(self) -> str:
        """Envelope de viabilidade EXPLICITO (mesmas regras do Guardian), com a
        aritmetica que o LLM precisa fazer antes de enviar. O dump YAML cru dos
        caps nao bastava: no smoke o modelo propos memory_gb*instances acima do
        total (invalid_recommendation), desperdicando rodadas. Fecha com um nudge
        de COBERTURA: o formato do executor e um continuo (muitos pequenos <->
        poucos grandes) e o modelo tende a ficar so no extremo scale-out."""
        c = self.cluster
        return (
            "## Restricoes rigidas de viabilidade (o Guardian REJEITA fora disto)\n"
            f"- executor_cores <= {c['max_cores_per_container']} (por container)\n"
            f"- executor_memory_gb <= {c['max_memory_gb_per_container']} (por container)\n"
            f"- executor_instances + 1 (driver) <= {c['max_pods']}\n"
            f"- executor_cores * executor_instances <= {c['total_cores']} (cores totais)\n"
            f"- executor_memory_gb * executor_instances + driver_memory_gb <= "
            f"{c['total_memory_gb']} (memoria total, GB)\n"
            "CALCULE os dois produtos (cores e memoria totais) ANTES de enviar; "
            "uma proposta inviavel desperdica uma rodada.\n"
            "COBERTURA: o formato do executor e um continuo -- desde MUITOS executores "
            "pequenos (ex.: 2c/7-8g x muitos) ate POUCOS executores grandes "
            "(ex.: 12c/48-96g x poucos). Cargas com muito shuffle/GC costumam preferir "
            "POUCOS executores com MAIS memoria cada. Nao fixe um unico formato: "
            "explore os dois extremos ao longo das iteracoes."
        )
