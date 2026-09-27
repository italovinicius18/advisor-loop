#!/usr/bin/env bash
# Fila in-cluster v4.1 (2026-09-27): sf500 sob envelope (hybrid3 + random) e a
# escada de ancoras em sf100 (A3 Sonnet, A0 heuristico, A1 Jev, A2 Haiku). 1 run por vez.
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
for name in sf500-v4-hybrid3 sf500-v4-random sf100-v4-a3-sonnet sf100-v4-a0-heuristic sf100-v4-a1-jev sf100-v4-a2-haiku; do
  pod=advisor-controller-$name
  echo "$(date -u +%FT%TZ) launching $pod"
  kubectl delete pod $pod -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  kubectl apply -f /queue/advisor-controller-$name.yaml
  sleep 60
  wait_pod "$pod"
done
echo "$(date -u +%FT%TZ) QUEUE DONE"
