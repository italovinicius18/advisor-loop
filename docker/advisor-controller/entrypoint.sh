#!/usr/bin/env bash
# Entrypoint do controller in-cluster: roda a campanha e SEMPRE sobe os
# artefatos para o MinIO (mesmo em falha/crash parcial), pra sobreviverem ao pod.
set -o pipefail
cd /opt/advisor-loop
CAMP="${CAMPAIGN:-campaigns/cluster-smoke-sf100.yaml}"

echo "[controller] iniciando campanha: ${CAMP} em $(date -u)"
python -u -m advisor_loop.runner run "${CAMP}" 2>&1 | tee /opt/advisor-loop/run.log
rc=$?
echo "[controller] campanha terminou rc=${rc} em $(date -u)"

echo "[controller] subindo artefatos para o MinIO..."
python -u /opt/advisor-loop/upload_results.py || echo "[controller] upload falhou (nao-fatal)"

# Durabilidade: despeja o journal no stdout para que sobreviva via `kubectl logs`
# mesmo que o upload ao MinIO falhe (o pod Completed nao permite exec/cp).
echo "[controller] ===JOURNAL DUMP (recuperavel via kubectl logs)==="
find /opt/advisor-loop/evidence -name 'journal.jsonl' -exec sh -c 'echo "--- $1 ---"; cat "$1"' _ {} \;
echo "[controller] ===END JOURNAL DUMP==="

echo "[controller] fim (rc=${rc})."
exit "${rc}"
