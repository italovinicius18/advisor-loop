#!/usr/bin/env bash
# Fila in-cluster de repeticoes (2026-10-05): A0 heuristico e A2 Haiku em sf100, r2 e r3.
# Espera a fila da regua top-k (pod advisor-queue) terminar: 1 run por vez.
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
while :; do
  ph=$(kubectl get pod advisor-queue -n $NS -o jsonpath='{.status.phase}' 2>/dev/null)
  [ "$ph" = "Succeeded" ] || [ "$ph" = "Failed" ] && break
  sleep 120
done
echo "$(date -u +%FT%TZ) advisor-queue terminou; iniciando repeticoes"
for name in sf100-v4-a0-heuristic-r2 sf100-v4-a2-haiku-r2 sf100-v4-a0-heuristic-r3 sf100-v4-a2-haiku-r3; do
  pod=advisor-controller-$name
  echo "$(date -u +%FT%TZ) launching $pod"
  kubectl delete pod $pod -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  kubectl apply -f /queue/advisor-controller-$name.yaml
  sleep 60
  wait_pod "$pod"
done
echo "$(date -u +%FT%TZ) QUEUE DONE"
