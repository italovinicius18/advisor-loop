# advisor-loop

LLM configuration-advisor episode harness for Spark on Kubernetes. An LLM
"advisor" recommends an executor config for a TPC-DS query against a
measured reference matrix (replay backend); episodes are journaled
idempotently so campaigns resume without re-spending on repeated advisor
calls. See `docs/spec.md` for the full design.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Configure

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=sk-or-...
```

## Import the reference matrix

Converts raw benchmark result JSON (e.g. from k8s-bigdata-bench Exp05) into
`data/reference.parquet`, the measured-cost source the replay backend reads:

```bash
.venv/bin/python scripts/import_reference.py <results_dir> data/reference.parquet
```

## Run a campaign

```bash
.venv/bin/advisor-loop run campaigns/pilot-replay.yaml
```

Prints a summary dict (`episodes_run`, `episodes_skipped`, `cost_usd`).
Re-running the same campaign is idempotent: episodes already journaled are
skipped *before* any advisor/API call, so no extra cost is incurred on
resume.

## Evidence

Each campaign run is journaled to `evidence/<campaign>/journal.jsonl`
(one JSON line per event). `advisor_loop.archiver.Archiver.publish(...)`
compresses the journal, writes a `summary.json`, scans for leaked secrets,
and commits under `evidence/<campaign>/`.

## Layout

```
advisor_loop/
  episode.py        episode/config data models
  journal.py         idempotent event journal
  guardian.py         infeasible/invalid-recommendation checks
  runner.py           campaign runner + CLI entrypoint
  supervisor.py        cross-episode orchestration
  archiver.py           evidence publishing + secret scan
  advisor/             LLM tool-loop client (OpenRouter)
  backends/replay.py    replay-from-reference execution backend
  knowledge/pack.py     knowledge pack fed to the advisor
  memory/store.py       cross-episode memory (optional)
  probes/                DuckDB probes + budget checks (local tunnel + in-cluster remote)
campaigns/            campaign YAML configs
service/              in-cluster DuckDB probe HTTP service (server + Dockerfile)
kubernetes/           k8s manifests (duckdb-probe-service Deployment/Service)
scripts/import_reference.py  reference-matrix importer
data/reference.parquet       measured cost matrix (imported)
evidence/                    per-campaign journals + summaries
```

## Data probing: local tunnel vs. in-cluster service

`advisor_loop.probes.duckdb_probe.DuckDBProbe` runs DuckDB locally against
MinIO over the SSH tunnel (`s3:` block in `datasets.yaml`) - fine for dev and
small scales, but every probed row streams through the tunnel.
`advisor_loop.probes.remote_probe.RemoteDuckDBProbe` talks instead to
`service/duckdb_probe_server.py`, a minimal DuckDB HTTP service meant to run
in-cluster next to MinIO (`kubernetes/duckdb-probe-service.yaml`): only small
SQL requests and small JSON results cross the tunnel. Both implement the same
probe interface, so `BudgetedProbe` wraps either unchanged; `wiring.py`
selects the remote client when `datasets.yaml` has a `probe_service.url`
block, and falls back to the local tunnel path otherwise (see the commented
example in `datasets.yaml`).
