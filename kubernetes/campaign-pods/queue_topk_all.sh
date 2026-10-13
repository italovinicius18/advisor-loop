#!/usr/bin/env bash
# Fila in-cluster regua top-3 completa no TPC-DS (2026-10-13): ranks 2 e 3 da matriz
# em SF10, 50, 250, 750 e 1000 (sf100/sf500 ja tem). Ordem de custo crescente. 1 run por vez.
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
for name in sf10-replay-r2 sf10-replay-r3 sf50-replay-r2 sf50-replay-r3 sf250-replay-r2 sf250-replay-r3 sf750-replay-r2 sf750-replay-r3 sf1000-replay-r2 sf1000-replay-r3; do
  pod=advisor-controller-$name
  echo "$(date -u +%FT%TZ) launching $pod"
  kubectl delete pod $pod -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  kubectl apply -f /queue/advisor-controller-$name.yaml
  sleep 60
  wait_pod "$pod"
done
echo "$(date -u +%FT%TZ) QUEUE DONE"
