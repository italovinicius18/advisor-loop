"""Campanhas das queries representativas de cada categoria do TPC-DS.

Categorias A (scan), B (shuffle) e C (subquery) vem do plano fisico (coluna
query_category da matriz). Regra de escolha, declarada antes de qualquer
resultado online: as duas queries mais proximas da mediana da categoria nos
atributos do plano (joins, exchanges, subconsultas, agregacoes, sorts,
profundidade), com oraculo em SF100 e SF500, fora as anomalias conhecidas
(q72, q95). As atuais q03 (A), q23 (B, caso extremo) e q50 (C) ja rodaram.

Bracos por SF: replay das 3 melhores celulas (regua) e o hibrido v4.1 com
ancora heuristica (A0) e Haiku 4.5 (A2), iguais aos da RQ4.

Uso: python scripts/gen_rep_campaigns.py --sfs 100 500 --image sf100-r27
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ["q19", "q71", "q94", "q66", "q05", "q02"]   # A: q19,q71  B: q94,q66  C: q05,q02
_spec = importlib.util.spec_from_file_location("gen_sf", Path(__file__).resolve().parent / "gen_sf_campaigns.py")
gen_sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_sf)

QUEUE_HEAD = """#!/usr/bin/env bash
# Fila in-cluster das queries representativas (A/B/C) em SF100 e SF500: regua
# top-3 + hibrido A0 heuristico e A2 Haiku. Gerado por scripts/gen_rep_campaigns.py.
# 1 run por vez; so disparar com o cluster livre.
set -u
NS=italo-unb
wait_pod() {
  local pod=$1 phase=""
  for _ in $(seq 1 1600); do
    phase=$(kubectl get pod "$pod" -n $NS -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$phase" = "Succeeded" ] || [ "$phase" = "Failed" ]; then
      echo "$(date -u +%FT%TZ) POD $pod phase=$phase"; return 0
    fi
    sleep 90
  done
  echo "$(date -u +%FT%TZ) TIMEOUT $pod (last=$phase)"; return 1
}
"""
QUEUE_LOOP = """for name in {names}; do
  pod=advisor-controller-$name
  echo "$(date -u +%FT%TZ) launching $pod"
  kubectl delete pod $pod -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  kubectl apply -f /queue/advisor-controller-$name.yaml
  sleep 60
  wait_pod "$pod"
done
echo "$(date -u +%FT%TZ) QUEUE DONE"
"""


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def generate(sfs: list[int], image: str, matrix: str | None = None, tries: int = 3) -> list[str]:
    matrix = matrix or str(Path(__file__).resolve().parents[1] / "data/matrix/training_data.parquet")
    names = []
    qlist = ", ".join(QUERIES)
    for sf in sfs:
        ds = f"sf{sf}"
        big = sf >= gen_sf.BIG_SF
        extra = ""
        if big:
            pin = ", ".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}"
                            for k, v in gen_sf.PINNED_BIG.items())
            extra = f"pinned: {{{pin}}}\nbaseline_config: {gen_sf.yaml_cfg(gen_sf.BASELINE_BIG)}\n"
        for rank in (1, 2, 3):
            cells = gen_sf.oracle_cells(matrix, sf, rank, QUERIES)
            cfgs = "\n".join(f"  {q}: {gen_sf.yaml_cfg(c)}   # t*={c['t_star_s']}s" for q, c in cells.items())
            name = f"{ds}-rep-replay" + ("" if rank == 1 else f"-r{rank}")
            _write(ROOT / f"campaigns/{name}.yaml", f"""# Regua (queries representativas): celula rank {rank} da matriz em {ds}. Gerado por scripts/gen_rep_campaigns.py.
name: {name}
mode: optimize
backend: spark_k8s
arm: fixed
level: L0
provider: oracle-replay
datasets: [{ds}]
queries: [{qlist}]
tries: {tries}
max_iters: 1
patience: 1
baseline: false
{extra}configs:
{cfgs}
""")
            names.append(name)
        for arm, anchor, prov in (("a0-heuristic", "heuristic", "none-heuristic"),
                                  ("a2-haiku", "llm", "anthropic/claude-haiku-4.5")):
            name = f"{ds}-rep-{arm}"
            _write(ROOT / f"campaigns/{name}.yaml", f"""# Hibrido v4.1, ancora {anchor}, queries representativas em {ds}. Gerado por scripts/gen_rep_campaigns.py.
name: {name}
mode: optimize
backend: spark_k8s
arm: hybrid
llm_every: 3
knobs: matrix3
level: L0
anchor: {anchor}
provider: {prov}
datasets: [{ds}]
queries: [{qlist}]
tries: {tries}
max_iters: 8
patience: 3
{extra}""")
            names.append(name)
    for name in names:
        _write(ROOT / f"kubernetes/campaign-pods/advisor-controller-{name}.yaml",
               gen_sf.POD.format(title=f"{name} (queries representativas; gerado por scripts/gen_rep_campaigns.py)",
                                 name=name, camp=name, image=image))
    _write(ROOT / "kubernetes/campaign-pods/queue_rep.sh", QUEUE_HEAD + QUEUE_LOOP.format(names=" ".join(names)))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfs", nargs="+", type=int, default=[100, 500])
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    print("fila:", " ".join(generate(a.sfs, a.image)))


if __name__ == "__main__":
    main()
