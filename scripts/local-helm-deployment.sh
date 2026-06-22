#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

COMMAND="${1:-up}"
NAMESPACE="${NAMESPACE:-llm-gpu-benchmarking}"
RELEASE="${RELEASE:-llm-gpu-benchmarking}"
INGRESS_CLASS="${INGRESS_CLASS:-traefik}"
INGRESS_HOST="${INGRESS_HOST:-llm-gpu-benchmarking.localhost}"
DELETE_NAMESPACE="${DELETE_NAMESPACE:-false}"

usage() {
  cat <<EOF
Usage: $0 [up|down|status|test|load]

Commands:
  up      Build the local image, install or upgrade the Helm release, and wait for health.
  down    Uninstall the Helm release. Set DELETE_NAMESPACE=true to remove the namespace.
  status  Show Helm release, Kubernetes workloads, and API health when reachable.
  test    Run one end-to-end benchmark workflow and wait for completion.
  load    Submit multiple benchmark workflows for dashboard and metrics validation.
EOF
}

load_env() {
  if [[ ! -f ".env" && -f ".env.example" ]]; then
    cp .env.example .env
    echo "Created .env from .env.example"
  fi

  set -a
  source .env
  set +a
}

load_runtime_env() {
  if [[ -z "${BASE_URL:-}" && -f ".runtime.env" ]]; then
    set -a
    source .runtime.env
    set +a
  fi

  BASE_URL="${BASE_URL:-http://${INGRESS_HOST}}"
}

python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
  elif [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
  elif [[ -x "venv/bin/python" ]]; then
    echo "venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo "python"
  fi
}

write_runtime_env() {
  cat > .runtime.env <<EOF
RUNTIME_MODE=kubernetes-helm
BASE_URL=http://${INGRESS_HOST}
KUBERNETES_NAMESPACE=${NAMESPACE}
HELM_RELEASE=${RELEASE}
INGRESS_CLASS=${INGRESS_CLASS}
INGRESS_HOST=${INGRESS_HOST}
EOF
  echo "Wrote .runtime.env for local Kubernetes Helm"
}

wait_for_api_health() {
  echo
  echo "API health:"
  for attempt in {1..20}; do
    if curl -fsS "http://${INGRESS_HOST}/health"; then
      echo
      write_runtime_env
      return 0
    fi
    sleep 2
    echo "Waiting for ingress health check (${attempt}/20)..."
  done

  echo "API health check failed at http://${INGRESS_HOST}/health" >&2
  return 1
}

up() {
  load_env

  kubectl cluster-info
  kubectl get nodes

  docker build -t llm-gpu-benchmarking-app:latest .

  kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

  if [[ -n "${NVIDIA_API_KEY:-}" ]]; then
    kubectl -n "${NAMESPACE}" create secret generic llm-gpu-benchmarking-secrets \
      --from-literal=NVIDIA_API_KEY="${NVIDIA_API_KEY}" \
      --dry-run=client -o yaml | kubectl apply -f -
  else
    echo "NVIDIA_API_KEY is not set; skipping optional hosted-validation secret."
  fi

  helm upgrade --install "${RELEASE}" helm/llm-gpu-benchmarking \
    --namespace "${NAMESPACE}" \
    --set ingress.enabled=true \
    --set "ingress.className=${INGRESS_CLASS}" \
    --set "ingress.hosts[0].host=${INGRESS_HOST}" \
    --set "ingress.hosts[0].paths[0].path=/" \
    --set "ingress.hosts[0].paths[0].pathType=Prefix" \
    --set-string "config.minimumTpsThreshold=${LLM_GPU_BENCHMARKING_MINIMUM_TPS_THRESHOLD:-100}" \
    --set-string "config.nimBaseUrl=${NIM_BASE_URL:-https://integrate.api.nvidia.com/v1}" \
    --set-string "config.nimTimeoutSeconds=${NIM_TIMEOUT_SECONDS:-30}" \
    --set-string "config.validationMode=${LLM_GPU_BENCHMARKING_VALIDATION_MODE:-validation}" \
    --set-string "config.logLevel=${LOG_LEVEL:-INFO}" \
    --set-string "worker.maxConcurrentActivities=${MAX_CONCURRENT_ACTIVITIES:-10}"

  kubectl -n "${NAMESPACE}" rollout status deployment \
    -l "app.kubernetes.io/instance=${RELEASE},app.kubernetes.io/component=api" \
    --timeout=120s
  kubectl -n "${NAMESPACE}" rollout status deployment \
    -l "app.kubernetes.io/instance=${RELEASE},app.kubernetes.io/component=worker" \
    --timeout=120s
  kubectl -n "${NAMESPACE}" rollout status deployment \
    -l "app.kubernetes.io/instance=${RELEASE},app.kubernetes.io/component=temporal" \
    --timeout=120s

  wait_for_api_health
}

down() {
  if helm status "${RELEASE}" --namespace "${NAMESPACE}" >/dev/null 2>&1; then
    helm uninstall "${RELEASE}" --namespace "${NAMESPACE}"
  else
    echo "Helm release '${RELEASE}' was not found in namespace '${NAMESPACE}'."
  fi

  if [[ "${DELETE_NAMESPACE}" == "true" ]]; then
    kubectl delete namespace "${NAMESPACE}" --ignore-not-found
  else
    echo "Namespace '${NAMESPACE}' was left in place."
    echo "Set DELETE_NAMESPACE=true to remove it."
  fi

  if [[ -f ".runtime.env" ]] && grep -q '^RUNTIME_MODE=kubernetes-helm$' .runtime.env; then
    rm .runtime.env
    echo "Removed Kubernetes Helm .runtime.env"
  fi
}

status() {
  echo "Helm release:"
  helm status "${RELEASE}" --namespace "${NAMESPACE}" || true

  echo
  echo "Kubernetes workloads:"
  kubectl -n "${NAMESPACE}" get deployments,services,ingress,pods \
    -l "app.kubernetes.io/instance=${RELEASE}" || true

  echo
  echo "API health:"
  curl -fsS "http://${INGRESS_HOST}/health" || true
  echo
}

test_run() {
  load_runtime_env

  MODEL_NAME="${MODEL_NAME:-Llama-3-70B}"
  TARGET_GPU="${TARGET_GPU:-H100-80GB}"
  TARGET_ENVIRONMENT="${TARGET_ENVIRONMENT:-kubernetes}"
  PRECISION_MODE="${PRECISION_MODE:-INT4}"
  TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"

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
  load_runtime_env

  REQUESTS="${REQUESTS:-30}"
  CONCURRENCY="${CONCURRENCY:-6}"
  POLL="${POLL:-true}"
  PYTHON_BIN="$(python_bin)"

  echo "Using BASE_URL=${BASE_URL}"
  "${PYTHON_BIN}" examples/load_test_validation_matrix.py \
    --base-url "${BASE_URL}" \
    --requests "${REQUESTS}" \
    --concurrency "${CONCURRENCY}" \
    --poll "${POLL}"
}

case "${COMMAND}" in
  up)
    up
    ;;
  down)
    down
    ;;
  status)
    status
    ;;
  test)
    test_run
    ;;
  load)
    load_run
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
