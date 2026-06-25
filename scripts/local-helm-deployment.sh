#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

COMMAND="${1:-up}"
NAMESPACE="${NAMESPACE:-llm-gpu-benchmarking}"
RELEASE="${RELEASE:-llm-gpu-benchmarking}"
INGRESS_CLASS="${INGRESS_CLASS:-traefik}"
INGRESS_HOST="${INGRESS_HOST:-llm-gpu-benchmarking.localhost}"

usage() {
  cat <<EOF
Usage: $0 [up|down|status]

Commands:
  up      Build the local image, install or upgrade the Helm release, and wait for health.
  down    Uninstall the Helm release and delete the namespace.
  status  Show Helm release, Kubernetes workloads, and API health when reachable.
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

  kubectl delete namespace "${NAMESPACE}" --ignore-not-found

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
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
