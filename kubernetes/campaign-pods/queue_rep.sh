#!/usr/bin/env bash
# Fila in-cluster das queries representativas (A/B/C) em SF100 e SF500: regua
# top-3 + hibrido A0 heuristico e A2 Haiku (2 execucoes cada). Gerado por
# scripts/gen_rep_campaigns.py. 1 run por vez. Antes de comecar, espera o cluster
# ficar OCIOSO: nenhum pod Running/Pending no namespace alem da infraestrutura
# fixa (MinIO, duckdb-probe, prometheus) e da propria fila, por IDLE_NEEDED
# checagens seguidas de 2 min (30 min). Assim nao disputa com outros trabalhos
# (SSCAD, exp05) nem dispara na pausa entre duas campanhas deles.
set -u
NS=italo-unb
IDLE_NEEDED=15
busy_pods() {
  kubectl get pods -n $NS --no-headers -o custom-columns=N:.metadata.name,P:.status.phase 2>/dev/null     | awk '$2=="Running" || $2=="Pending" {print $1}'     | grep -v -E '^(minio-|duckdb-probe-|prometheus-|advisor-queue)'
}
wait_cluster_idle() {
  local idle=0 busy
  echo "$(date -u +%FT%TZ) esperando o cluster ficar ocioso ($IDLE_NEEDED checagens seguidas)"
  while [ $idle -lt $IDLE_NEEDED ]; do
    busy=$(busy_pods)
    if [ -z "$busy" ]; then idle=$((idle+1)); else
      [ $idle -gt 0 ] && echo "$(date -u +%FT%TZ) ocupado de novo: $(echo $busy | head -c 200)"
      idle=0
    fi
    sleep 120
  done
  echo "$(date -u +%FT%TZ) cluster ocioso ha 30 min: iniciando a fila"
}
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
wait_cluster_idle
for name in sf100-rep-replay sf100-rep-replay-r2 sf100-rep-replay-r3 sf100-rep-a0-heuristic sf100-rep-a2-haiku sf100-rep-a0-heuristic-r2 sf100-rep-a2-haiku-r2 sf500-rep-replay sf500-rep-replay-r2 sf500-rep-replay-r3 sf500-rep-a0-heuristic sf500-rep-a2-haiku sf500-rep-a0-heuristic-r2 sf500-rep-a2-haiku-r2; do
  pod=advisor-controller-$name
  echo "$(date -u +%FT%TZ) launching $pod"
  kubectl delete pod $pod -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  kubectl apply -f /queue/advisor-controller-$name.yaml
  sleep 60
  wait_pod "$pod"
done
echo "$(date -u +%FT%TZ) QUEUE DONE"
