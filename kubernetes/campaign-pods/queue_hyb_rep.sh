#!/usr/bin/env bash
# Fila de repeticoes do hibrido v3.1 em SF750 e SF1000 (2026-10-15): n=2 nos SFs
# grandes, onde o SF750 ficou perto de zero (-0.04). 1 run por vez.
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
for name in sf750-hybrid3-r2 sf1000-hybrid3-r2; do
  pod=advisor-controller-$name
  echo "$(date -u +%FT%TZ) launching $pod"
  kubectl delete pod $pod -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  kubectl apply -f /queue/advisor-controller-$name.yaml
  sleep 60
  wait_pod "$pod"
done
echo "$(date -u +%FT%TZ) QUEUE DONE"
