from pathlib import Path
import yaml
from advisor_loop.advisor.provider import OpenRouterClient
from advisor_loop.backends.replay import ReplayBackend
from advisor_loop.guardian import Guardian
from advisor_loop.journal import Journal
from advisor_loop.knowledge.pack import KnowledgePack
from advisor_loop.memory.store import MemoryStore
from advisor_loop.probes.budget import BudgetedProbe
from advisor_loop.probes.duckdb_probe import DuckDBProbe
from advisor_loop.runner import Deps

def build_deps(campaign_path: Path, root: Path = Path(".")) -> Deps:
    camp = yaml.safe_load(campaign_path.read_text())
    ev = root / "evidence" / camp["name"]
    pack = KnowledgePack.load(root / "packs")
    tables_by_dataset = {}
    ds_file = root / "datasets.yaml"
    if ds_file.exists():
        tables_by_dataset = yaml.safe_load(ds_file.read_text()) or {}

    def probe_factory(ep):
        tables = tables_by_dataset.get(ep.dataset)
        return BudgetedProbe(DuckDBProbe(tables)) if tables else None

    return Deps(client=OpenRouterClient(), journal=Journal(ev / "journal.jsonl"),
                pack=pack, guardian=Guardian(pack.cluster),
                backend=ReplayBackend(root / "data" / "reference.parquet"),
                memory=MemoryStore(ev / "memory.jsonl"), probe_factory=probe_factory)
