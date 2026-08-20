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
        return "\n\n".join(parts)
