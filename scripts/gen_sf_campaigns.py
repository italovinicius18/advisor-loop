"""Gera as campanhas multi-SF (replay do oraculo + hybrid3 + random) e os pods.

Uso: python scripts/gen_sf_campaigns.py --sfs 10 50 250 500 --image sf100-r14
Le a matriz para a celula vencedora (min duration_ms success) de cada (sf, query)
e escreve campaigns/sf<N>-{replay,hybrid3,random}.yaml + kubernetes/campaign-pods/
advisor-controller-sf<N>-{replay,hybrid3,random}.yaml. Knobs extras pinados nos
defaults da matriz (validos para SF<=500; SF750 usou sp=2000/driver 16g -- fora).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import duckdb

QUERIES = ["q03", "q23", "q50"]
PINNED = dict(driver_memory_gb=8, shuffle_partitions=200, advisory_partition_mb=64,
              aqe_enabled=True, broadcast_threshold_mb=10)
ROOT = Path(__file__).resolve().parents[1]

POD = """# {title}
apiVersion: v1
kind: Pod
metadata:
  name: advisor-controller-{name}
  namespace: italo-unb
  labels:
    app: advisor-controller
    campaign: {camp}
spec:
  serviceAccountName: ns-admin
  restartPolicy: Never
  imagePullSecrets:
    - name: ghcr-pull
  nodeSelector:
    kubernetes.io/hostname: ids-pb
  containers:
    - name: controller
      image: ghcr.io/italovinicius18/advisor-controller:{image}
      imagePullPolicy: Always
      env:
        - name: CAMPAIGN
          value: campaigns/{camp}.yaml
        - name: RESULT_PREFIX
          value: {camp}
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: advisor-secrets
              key: OPENROUTER_API_KEY
      resources:
        requests: {{memory: "512Mi", cpu: "250m"}}
        limits: {{memory: "1Gi", cpu: "500m"}}
"""


def oracle_cells(matrix: str, sf: int) -> dict[str, dict]:
    con = duckdb.connect()
    out = {}
    for q in QUERIES:
        row = con.execute(
            """SELECT executor_cores, CAST(executor_memory_gb AS INT), workers, duration_ms
               FROM read_parquet(?) WHERE status='success' AND CAST(scale_factor AS INT)=? AND query=?
               ORDER BY duration_ms LIMIT 1""", [matrix, sf, q]).fetchone()
        if row is None:
            raise SystemExit(f"sem celula success para sf{sf} {q}")
        c, m, w, d = row
        out[q] = dict(executor_cores=int(c), executor_memory_gb=int(m), executor_instances=int(w),
                      t_star_s=round(d / 1000, 1), **PINNED)
    con.close()
    return out


def yaml_cfg(c: dict) -> str:
    return ("{executor_cores: %d, executor_memory_gb: %d, executor_instances: %d, driver_memory_gb: %d, "
            "shuffle_partitions: %d, advisory_partition_mb: %d, aqe_enabled: %s, broadcast_threshold_mb: %d}"
            % (c["executor_cores"], c["executor_memory_gb"], c["executor_instances"], c["driver_memory_gb"],
               c["shuffle_partitions"], c["advisory_partition_mb"], str(c["aqe_enabled"]).lower(),
               c["broadcast_threshold_mb"]))


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print("wrote", path.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfs", nargs="+", type=int, required=True)
    ap.add_argument("--image", required=True, help="tag do advisor-controller (ex.: sf100-r14)")
    ap.add_argument("--matrix", default=str(ROOT / "data/matrix/training_data.parquet"))
    ap.add_argument("--tries", type=int, default=3)
    a = ap.parse_args()
    queue = []
    for sf in a.sfs:
        ds = f"sf{sf}"
        cells = oracle_cells(a.matrix, sf)
        cfgs = "\n".join(f"  {q}: {yaml_cfg(c)}   # t*={c['t_star_s']}s" for q, c in cells.items())
        write(ROOT / f"campaigns/{ds}-replay.yaml", f"""# Replay do oraculo em {ds}: celula vencedora da matriz por query, no harness
# quente (regua t*_harness deste SF). Gerado por scripts/gen_sf_campaigns.py.
name: {ds}-replay
mode: optimize
backend: spark_k8s
arm: fixed
level: L0
provider: oracle-replay
datasets: [{ds}]
queries: [{', '.join(QUERIES)}]
tries: {a.tries}
max_iters: 1
patience: 1
configs:
{cfgs}
""")
        write(ROOT / f"campaigns/{ds}-hybrid3.yaml", f"""# Hibrido v3.1 (consolidacao, knobs pinados) em {ds}. Gerado por scripts/gen_sf_campaigns.py.
name: {ds}-hybrid3
mode: optimize
backend: spark_k8s
arm: hybrid
llm_every: 3
knobs: matrix3
level: L0
provider: anthropic/claude-sonnet-5
datasets: [{ds}]
queries: [{', '.join(QUERIES)}]
tries: {a.tries}
max_iters: 8
patience: 3
""")
        write(ROOT / f"campaigns/{ds}-random.yaml", f"""# Controle random-8 (schema livre) em {ds}. Gerado por scripts/gen_sf_campaigns.py.
name: {ds}-random
mode: optimize
backend: spark_k8s
arm: random
seed: 1
level: L0
provider: random-free-8
datasets: [{ds}]
queries: [{', '.join(QUERIES)}]
tries: {a.tries}
max_iters: 8
patience: 3
""")
        for arm in ("replay", "hybrid3", "random"):
            name = f"{ds}-{arm}"
            write(ROOT / f"kubernetes/campaign-pods/advisor-controller-{name}.yaml",
                  POD.format(title=f"{name} (multi-SF; gerado por scripts/gen_sf_campaigns.py)",
                             name=name, camp=name, image=a.image))
            queue.append(name)
    print("ordem sugerida da fila:", " ".join(queue))


if __name__ == "__main__":
    main()
