import json
from pathlib import Path
from advisor_loop.episode import Episode, Outcome, Recommendation

class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def distill(self, ep: Episode, rec: Recommendation, out: Outcome) -> dict:
        qstat = {q: r.status for q, r in out.per_query.items()}
        mem = {"workload": ep.workload, "dataset": ep.dataset, "queries": ep.queries,
               "config": rec.config.model_dump(), "status": out.status,
               "lesson": f"[{ep.dataset} {','.join(ep.queries)}] config "
                         f"{rec.config.executor_cores}c/{rec.config.executor_memory_gb}g"
                         f"x{rec.config.executor_instances} -> {out.status} ({qstat})"}
        with open(self.path, "a") as f:
            f.write(json.dumps(mem, ensure_ascii=False) + "\n")
        return mem

    def retrieve(self, ep: Episode, k: int = 5) -> list[str]:
        def leaks(m):
            # loso-scale:<sf>: fold isola por dataset inteiro -> vazamento se
            # QUALQUER memoria do mesmo dataset entra (independente da query).
            if ep.split.startswith("loso-scale:"):
                return m["dataset"] == ep.dataset
            # loso-query:<q>: fold isola por query inteira -> vazamento se
            # QUALQUER memoria com overlap de query entra (independente do dataset).
            if ep.split.startswith("loso-query:"):
                return bool(set(m["queries"]) & set(ep.queries))
            # default ("none" ou qualquer outro valor): anti-vazamento par a par -
            # nada do proprio (dataset, query) exato.
            return m["dataset"] == ep.dataset and bool(set(m["queries"]) & set(ep.queries))
        def score(m):
            return (m["workload"] == ep.workload, m["dataset"] == ep.dataset)
        pool = [m for m in self._all() if not leaks(m)]
        pool.sort(key=score, reverse=True)
        return [m["lesson"] for m in pool[:k]]
