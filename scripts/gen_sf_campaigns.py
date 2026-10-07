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
# TPC-H (exp06): uma query por choke point (Boncz et al., TPCTC 2013) --
# q01 agregacao/scan (leve), q09 multi-join (pesada), q18 group-by grande
# com subquery (pesada). Dataset "tpch-sf<N>" (advisor_loop/benchmarks.py).
BENCH = {"tpcds": dict(queries=QUERIES, prefix="", matrix="data/matrix/training_data.parquet"),
         "tpch": dict(queries=["q01", "q09", "q18"], prefix="tpch-",
                      matrix="data/matrix/tpch_training_data.parquet")}
PINNED = dict(driver_memory_gb=8, shuffle_partitions=200, advisory_partition_mb=64,
              aqe_enabled=True, broadcast_threshold_mb=10)
# SF>=750: a matriz so rodou a grade 'large' (12c x 48g x W2/W4/W6) e SO com o
# patch v2 (shuffle_partitions=2000 + advisory 64m + driver 16g); o default
# Spark (8c x 1w x 8g, driver 4g) nao completa. Logo: knobs pinados = os da
# matriz nesse SF, baseline = a menor celula large (W2), e SEM braco random
# (configs pequenas sorteadas viram timeouts de 200 min cada).
PINNED_BIG = dict(driver_memory_gb=16, shuffle_partitions=2000, advisory_partition_mb=64,
                  aqe_enabled=True, broadcast_threshold_mb=10)
BASELINE_BIG = dict(executor_cores=12, executor_memory_gb=48, executor_instances=2, **PINNED_BIG)
BIG_SF = 750
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


def pinned_for(sf: int) -> dict:
    return PINNED_BIG if sf >= BIG_SF else PINNED


def oracle_cells(matrix: str, sf: int, rank: int = 1, queries=None) -> dict[str, dict]:
    """rank=1 = celula vencedora (t*); rank=k = k-esima CELULA distinta,
    ordenada pelo melhor try (mesma regua do t*)."""
    con = duckdb.connect()
    out = {}
    pinned = pinned_for(sf)
    for q in (queries or QUERIES):
        row = con.execute(
            """SELECT executor_cores, CAST(executor_memory_gb AS INT), workers, min(duration_ms) AS d
               FROM read_parquet(?) WHERE status='success' AND CAST(scale_factor AS INT)=? AND query=?
               GROUP BY 1, 2, 3 ORDER BY d LIMIT 1 OFFSET ?""", [matrix, sf, q, rank - 1]).fetchone()
        if row is None:
            raise SystemExit(f"sem celula success rank {rank} para sf{sf} {q}")
        c, m, w, d = row
        out[q] = dict(executor_cores=int(c), executor_memory_gb=int(m), executor_instances=int(w),
                      t_star_s=round(d / 1000, 1), **pinned)
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
    ap.add_argument("--benchmark", choices=sorted(BENCH), default="tpcds")
    ap.add_argument("--matrix", default=None, help="default: matriz do benchmark")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--rank", type=int, default=1,
                    help="k>1 gera SO o replay da k-esima celula (sf<N>-replay-r<k>), p/ a regua top-k")
    a = ap.parse_args()
    bench = BENCH[a.benchmark]
    qs = bench["queries"]
    matrix = a.matrix or str(ROOT / bench["matrix"])
    queue = []
    for sf in a.sfs:
        ds = f"{bench['prefix']}sf{sf}"
        big = sf >= BIG_SF
        cells = oracle_cells(matrix, sf, a.rank, qs)
        replay = f"{ds}-replay" if a.rank == 1 else f"{ds}-replay-r{a.rank}"
        # ranks extras so alimentam a regua: sem baseline default (no sf500 ele
        # domina o custo do replay).
        # No TPC-H nem o rank 1 precisa: o replay fica fora do envelope, entao o
        # baseline seria o default (8c x 1w x 8g) -- horas em q09/q18 no sf500.
        nobase = "baseline: false\n" if (a.rank > 1 or a.benchmark != "tpcds") else ""
        cfgs = "\n".join(f"  {q}: {yaml_cfg(c)}   # t*={c['t_star_s']}s" for q, c in cells.items())
        extra = ""
        if big:
            pin = ", ".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}"
                            for k, v in PINNED_BIG.items())
            extra = (f"# SF>={BIG_SF}: knobs pinados = os da matriz neste SF (patch v2) e baseline =\n"
                     f"# menor celula 'large' (12c x 2w x 48g); o default Spark nao roda aqui.\n"
                     f"pinned: {{{pin}}}\nbaseline_config: {yaml_cfg(BASELINE_BIG)}\n")
        write(ROOT / f"campaigns/{replay}.yaml", f"""# Replay do oraculo em {ds}: celula rank {a.rank} da matriz por query, no harness
# quente (regua t*_harness deste SF = min dos ranks). Gerado por scripts/gen_sf_campaigns.py.
name: {replay}
mode: optimize
backend: spark_k8s
arm: fixed
level: L0
provider: oracle-replay
datasets: [{ds}]
queries: [{', '.join(qs)}]
tries: {a.tries}
max_iters: 1
patience: 1
{nobase}{extra}configs:
{cfgs}
""")
        if a.rank > 1:
            write(ROOT / f"kubernetes/campaign-pods/advisor-controller-{replay}.yaml",
                  POD.format(title=f"{replay} (regua top-k; gerado por scripts/gen_sf_campaigns.py)",
                             name=replay, camp=replay, image=a.image))
            queue.append(replay)
            continue
        if a.benchmark == "tpcds":
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
queries: [{', '.join(qs)}]
tries: {a.tries}
max_iters: 8
patience: 3
{extra}""")
            hybrids = ("hybrid3",)
        else:
            hybrids = ("a2-haiku", "a0-heuristic")
            for arm, anchor, prov in (("a2-haiku", "llm", "anthropic/claude-haiku-4.5"),
                                      ("a0-heuristic", "heuristic", "none-heuristic")):
                write(ROOT / f"campaigns/{ds}-{arm}.yaml", f"""# Hibrido v4.1 (envelope de escala), ancora {anchor}, em {ds}. Gerado por scripts/gen_sf_campaigns.py.
name: {ds}-{arm}
mode: optimize
backend: spark_k8s
arm: hybrid
llm_every: 3
knobs: matrix3
level: L0
anchor: {anchor}
provider: {prov}
datasets: [{ds}]
queries: [{', '.join(qs)}]
tries: {a.tries}
max_iters: 8
patience: 3
{extra}""")
        arms = ("replay", *hybrids) + (() if big else ("random",))
        if big:
            print(f"{ds}: sem braco random (SF>={BIG_SF})")
        if not big:
            write(ROOT / f"campaigns/{ds}-random.yaml", f"""# Controle random-8 (schema livre) em {ds}. Gerado por scripts/gen_sf_campaigns.py.
name: {ds}-random
mode: optimize
backend: spark_k8s
arm: random
seed: 1
level: L0
provider: random-free-8
datasets: [{ds}]
queries: [{', '.join(qs)}]
tries: {a.tries}
max_iters: 8
patience: 3
""")
        for arm in arms:
            name = f"{ds}-{arm}"
            write(ROOT / f"kubernetes/campaign-pods/advisor-controller-{name}.yaml",
                  POD.format(title=f"{name} (multi-SF; gerado por scripts/gen_sf_campaigns.py)",
                             name=name, camp=name, image=a.image))
            queue.append(name)
    print("ordem sugerida da fila:", " ".join(queue))


if __name__ == "__main__":
    main()
