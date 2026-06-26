#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

INGRESS_HOST="${INGRESS_HOST:-llm-gpu-benchmarking.localhost}"
COMMAND="${1:-load}"

if [[ -z "${BASE_URL:-}" && -f ".runtime.env" ]]; then
  set -a
  source .runtime.env
  set +a
fi

BASE_URL="${BASE_URL:-http://${INGRESS_HOST}}"

usage() {
  cat <<EOF
Usage: $0 [load|test]

Commands:
  load    Submit multiple benchmark workflows for dashboard and metrics validation.
  test    Run one end-to-end benchmark workflow and wait for completion.
EOF
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  python_bin="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
else
  python_bin="python"
fi

test_run() {
  MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}"
  TARGET_GPU="${TARGET_GPU:-A10G-24GB}"
  TARGET_ENVIRONMENT="${TARGET_ENVIRONMENT:-kubernetes}"
  PRECISION_MODE="${PRECISION_MODE:-INT4}"
  TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"

  echo "Using BASE_URL=${BASE_URL}"
  echo "Health check: ${BASE_URL}/health"
  curl -fsS "${BASE_URL}/health" | jq .

  echo "Submitting benchmark workflow"
  response="$(curl -fsS -X POST "${BASE_URL}/benchmarks" \
    -H "Content-Type: application/json" \
    -d "{\"model_name\":\"${MODEL_NAME}\",\"target_gpu\":\"${TARGET_GPU}\",\"target_environment\":\"${TARGET_ENVIRONMENT}\",\"precision_mode\":\"${PRECISION_MODE}\"}")"
  echo "${response}" | jq .

  workflow_id="$(echo "${response}" | jq -r '.workflow_id')"
  deadline=$((SECONDS + TIMEOUT_SECONDS))

  while (( SECONDS < deadline )); do
    status_response="$(curl -fsS "${BASE_URL}/benchmarks/${workflow_id}")"
    status="$(echo "${status_response}" | jq -r '.status')"
    final_status="$(echo "${status_response}" | jq -r '.pipeline_summary.final_status // empty')"
    echo "workflow=${workflow_id} temporal_status=${status} final_status=${final_status:-pending}"

    if [[ "${status}" == "COMPLETED" ]]; then
      echo "${status_response}" | jq .
      if [[ "${final_status}" == "Model_Service_Ready_To_Deploy" ]]; then
        return 0
      fi
      echo "Benchmark workflow completed with final_status=${final_status}" >&2
      return 1
    fi

    sleep 2
  done

  echo "Timed out waiting for workflow ${workflow_id}" >&2
  return 1
}

load_run() {
  REQUESTS="${REQUESTS:-5}"
  CONCURRENCY="${CONCURRENCY:-1}"
  POLL="${POLL:-true}"

  echo "Using BASE_URL=${BASE_URL}"
  "${python_bin}" examples/load_test_validation_matrix.py \
    --base-url "${BASE_URL}" \
    --requests "${REQUESTS}" \
    --concurrency "${CONCURRENCY}" \
    --poll "${POLL}"
}

case "${COMMAND}" in
  load)
    load_run
    ;;
  test)
    test_run
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
