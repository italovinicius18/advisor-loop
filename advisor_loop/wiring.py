from pathlib import Path
import yaml
from advisor_loop.advisor.provider import OpenRouterClient
from advisor_loop.backends.proposal import ProposalBackend
from advisor_loop.backends.replay import ReplayBackend
from advisor_loop.backends.spark_k8s import SparkK8sBackend
from advisor_loop.backends.spark_submitter import SparkSubmitter
from advisor_loop.guardian import Guardian
from advisor_loop.journal import Journal
from advisor_loop.knowledge.pack import KnowledgePack
from advisor_loop.memory.store import MemoryStore
from advisor_loop.probes.budget import BudgetedProbe
from advisor_loop.probes.duckdb_probe import DuckDBProbe
from advisor_loop.probes.remote_probe import RemoteDuckDBProbe
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
    probe_service = ds_config.get("probe_service")
    spark_config = ds_config.get("spark")
    tables_by_dataset = {k: v for k, v in ds_config.items()
                        if k not in ("s3", "probe_service", "spark")}

    def probe_factory(ep):
        tables = tables_by_dataset.get(ep.dataset)
        if not tables:
            return None
        # probe_service: push compute to the data - an in-cluster DuckDB HTTP
        # service (kubernetes/duckdb-probe-service.yaml) runs next to MinIO
        # and small SQL/JSON crosses the tunnel instead of table data. Falls
        # back to the local DuckDBProbe-over-tunnel path when unset.
        if probe_service and probe_service.get("url"):
            return BudgetedProbe(RemoteDuckDBProbe(probe_service["url"], tables))
        return BudgetedProbe(DuckDBProbe(tables, s3_config=s3_config))

    if camp.get("backend") == "spark_k8s":
        backend = SparkK8sBackend(SparkSubmitter(
            ssh_cmd=spark_config["ssh_cmd"],
            image=spark_config["image"],
            node_selector=spark_config.get("node_selector"),
            pod_template_file=spark_config.get("pod_template_file"),
            node_affinity_hostnames=spark_config.get("node_affinity_hostnames")))
    elif camp.get("backend") == "proposal":
        backend = ProposalBackend()
    else:
        backend = ReplayBackend(root / "data" / "reference.parquet")

    return Deps(client=OpenRouterClient(), journal=Journal(ev / "journal.jsonl"),
                pack=pack, guardian=Guardian(pack.cluster),
                backend=backend,
                memory=MemoryStore(ev / "memory.jsonl"), probe_factory=probe_factory)
