#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${BASE_URL:-}" && -f ".runtime.env" ]]; then
  set -a
  source .runtime.env
  set +a
fi

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL_NAME="${MODEL_NAME:-Llama-3-70B}"
TARGET_GPU="${TARGET_GPU:-H100-80GB}"
TARGET_ENVIRONMENT="${TARGET_ENVIRONMENT:-kubernetes}"
PRECISION_MODE="${PRECISION_MODE:-INT4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"

echo "Using BASE_URL=${BASE_URL}"
echo "Health check: ${BASE_URL}/health"
curl -fsS "${BASE_URL}/health" | jq .

echo "Submitting smoke workflow"
response="$(curl -fsS -X POST "${BASE_URL}/factory/ingest" \
  -H "Content-Type: application/json" \
  -d "{\"model_name\":\"${MODEL_NAME}\",\"target_gpu\":\"${TARGET_GPU}\",\"target_environment\":\"${TARGET_ENVIRONMENT}\",\"precision_mode\":\"${PRECISION_MODE}\"}")"
echo "${response}" | jq .

workflow_id="$(echo "${response}" | jq -r '.workflow_id')"
deadline=$((SECONDS + TIMEOUT_SECONDS))

while (( SECONDS < deadline )); do
  status_response="$(curl -fsS "${BASE_URL}/factory/status/${workflow_id}")"
  status="$(echo "${status_response}" | jq -r '.status')"
  final_status="$(echo "${status_response}" | jq -r '.pipeline_summary.final_status // empty')"
  echo "workflow=${workflow_id} temporal_status=${status} final_status=${final_status:-pending}"

  if [[ "${status}" == "COMPLETED" ]]; then
    echo "${status_response}" | jq .
    if [[ "${final_status}" == "Model_Service_Ready_To_Deploy" ]]; then
      exit 0
    fi
    echo "Smoke workflow completed with final_status=${final_status}" >&2
    exit 1
  fi

  sleep 2
done

echo "Timed out waiting for workflow ${workflow_id}" >&2
exit 1
