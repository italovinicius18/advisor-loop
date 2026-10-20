#!/usr/bin/env bash
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
for name in sf100-rep-replay sf100-rep-replay-r2 sf100-rep-replay-r3 sf100-rep-a0-heuristic sf100-rep-a2-haiku sf500-rep-replay sf500-rep-replay-r2 sf500-rep-replay-r3 sf500-rep-a0-heuristic sf500-rep-a2-haiku; do
  pod=advisor-controller-$name
  echo "$(date -u +%FT%TZ) launching $pod"
  kubectl delete pod $pod -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  kubectl apply -f /queue/advisor-controller-$name.yaml
  sleep 60
  wait_pod "$pod"
done
echo "$(date -u +%FT%TZ) QUEUE DONE"
