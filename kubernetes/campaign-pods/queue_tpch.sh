#!/usr/bin/env bash
# Fila in-cluster TPC-H online (2026-10-07): regua top-3 + escada A2 Haiku / A0 heuristica
# em sf500 e sf750. 1 run por vez. tpch-sf500-random foi PULADO em 2026-10-09 (decisao do
# usuario: ~1 dia de cluster, nao muda RQ3/RQ4); a fila em execucao recebeu um pod no-op.
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
for name in tpch-sf500-replay tpch-sf500-replay-r2 tpch-sf500-replay-r3 tpch-sf500-a2-haiku tpch-sf500-a0-heuristic tpch-sf750-replay tpch-sf750-replay-r2 tpch-sf750-replay-r3 tpch-sf750-a2-haiku tpch-sf750-a0-heuristic; do
  pod=advisor-controller-$name
  echo "$(date -u +%FT%TZ) launching $pod"
  kubectl delete pod $pod -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  kubectl apply -f /queue/advisor-controller-$name.yaml
  sleep 60
  wait_pod "$pod"
done
echo "$(date -u +%FT%TZ) QUEUE DONE"
