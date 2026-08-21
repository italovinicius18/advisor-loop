from pathlib import Path
import yaml
from advisor_loop.advisor.provider import OpenRouterClient
from advisor_loop.backends.proposal import ProposalBackend
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

    # datasets.yaml: optional top-level `s3:` block (MinIO/S3 creds) + per-dataset
    # table maps. Backward-compat: sem bloco s3, DuckDBProbe(tables) como antes.
    ds_config = {}
    ds_file = root / "datasets.yaml"
    if ds_file.exists():
        ds_config = yaml.safe_load(ds_file.read_text()) or {}
    s3_config = ds_config.get("s3")
    tables_by_dataset = {k: v for k, v in ds_config.items() if k != "s3"}

    def probe_factory(ep):
        tables = tables_by_dataset.get(ep.dataset)
        return BudgetedProbe(DuckDBProbe(tables, s3_config=s3_config)) if tables else None

    backend = (ProposalBackend() if camp.get("backend") == "proposal"
               else ReplayBackend(root / "data" / "reference.parquet"))

    return Deps(client=OpenRouterClient(), journal=Journal(ev / "journal.jsonl"),
                pack=pack, guardian=Guardian(pack.cluster),
                backend=backend,
                memory=MemoryStore(ev / "memory.jsonl"), probe_factory=probe_factory)
