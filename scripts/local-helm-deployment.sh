#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

COMMAND="${1:-up}"
NAMESPACE="${NAMESPACE:-llm-gpu-benchmarking}"
RELEASE="${RELEASE:-llm-gpu-benchmarking}"
INGRESS_CLASS="${INGRESS_CLASS:-traefik}"
INGRESS_HOST="${INGRESS_HOST:-llm-gpu-benchmarking.localhost}"
MONITORING_ENABLED="${MONITORING_ENABLED:-true}"
GRAFANA_HOST="${GRAFANA_HOST:-grafana.llm-gpu-benchmarking.localhost}"
PROMETHEUS_HOST="${PROMETHEUS_HOST:-prometheus.llm-gpu-benchmarking.localhost}"

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
MONITORING_ENABLED=${MONITORING_ENABLED}
GRAFANA_URL=http://${GRAFANA_HOST}
PROMETHEUS_URL=http://${PROMETHEUS_HOST}
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
    --set "monitoring.enabled=${MONITORING_ENABLED}" \
    --set "monitoring.grafana.ingress.enabled=${MONITORING_ENABLED}" \
    --set "monitoring.grafana.ingress.className=${INGRESS_CLASS}" \
    --set "monitoring.grafana.ingress.host=${GRAFANA_HOST}" \
    --set "monitoring.prometheus.ingress.enabled=${MONITORING_ENABLED}" \
    --set "monitoring.prometheus.ingress.className=${INGRESS_CLASS}" \
    --set "monitoring.prometheus.ingress.host=${PROMETHEUS_HOST}" \
    --set-string "monitoring.grafana.adminUser=${GRAFANA_ADMIN_USER:-admin}" \
    --set-string "monitoring.grafana.adminPassword=${GRAFANA_ADMIN_PASSWORD:-admin}" \
    --set-string "config.minimumTpsThreshold=${LLM_GPU_BENCHMARKING_MINIMUM_TPS_THRESHOLD:-100}" \
    --set-string "config.nimBaseUrl=${NIM_BASE_URL:-https://integrate.api.nvidia.com/v1}" \
    --set-string "config.nimTimeoutSeconds=${NIM_TIMEOUT_SECONDS:-30}" \
    --set-string "config.validationMode=${LLM_GPU_BENCHMARKING_VALIDATION_MODE:-local}" \
    --set-string "config.localLlmRoot=${LLM_GPU_BENCHMARKING_LOCAL_LLM_ROOT:-.runtime/local-llm}" \
    --set-string "config.localLlmProfilesPath=${LLM_GPU_BENCHMARKING_LOCAL_LLM_PROFILES_PATH:-src/config/local_llm_profiles.json}" \
    --set-string "config.llamaCppRepoUrl=${LLAMA_CPP_REPO_URL:-https://github.com/ggml-org/llama.cpp.git}" \
    --set-string "config.llamaCppGitRef=${LLAMA_CPP_GIT_REF:-master}" \
    --set-string "config.llamaCppCmakeArgs=${LLAMA_CPP_CMAKE_ARGS:--DGGML_METAL=OFF -DBUILD_SHARED_LIBS=OFF}" \
    --set-string "config.llamaCppBuildJobs=${LLAMA_CPP_BUILD_JOBS:-2}" \
    --set-string "config.llamaCppThreads=${LLAMA_CPP_THREADS:-2}" \
    --set-string "config.llamaCppBenchRepetitions=${LLAMA_CPP_BENCH_REPETITIONS:-1}" \
    --set-string "config.localLlmDownloadTimeoutSeconds=${LOCAL_LLM_DOWNLOAD_TIMEOUT_SECONDS:-1800}" \
    --set-string "config.localLlmBenchmarkTimeoutSeconds=${LOCAL_LLM_BENCHMARK_TIMEOUT_SECONDS:-900}" \
    --set-string "config.localLlmWarmupOnStartup=${LOCAL_LLM_WARMUP_ON_STARTUP:-true}" \
    --set-string "config.localLlmWarmupModels=${LOCAL_LLM_WARMUP_MODELS:-all}" \
    --set-string "config.localLlmWarmupPrecisionModes=${LOCAL_LLM_WARMUP_PRECISION_MODES:-INT4}" \
    --set-string "config.localLlmWarmupRequired=${LOCAL_LLM_WARMUP_REQUIRED:-true}" \
    --set-string "config.logLevel=${LOG_LEVEL:-INFO}" \
    --set-string "worker.maxConcurrentActivities=${MAX_CONCURRENT_ACTIVITIES:-1}"

  kubectl -n "${NAMESPACE}" rollout status deployment \
    -l "app.kubernetes.io/instance=${RELEASE},app.kubernetes.io/component=api" \
    --timeout=120s
  kubectl -n "${NAMESPACE}" rollout status deployment \
    -l "app.kubernetes.io/instance=${RELEASE},app.kubernetes.io/component=worker" \
    --timeout=120s
  kubectl -n "${NAMESPACE}" rollout status deployment \
    -l "app.kubernetes.io/instance=${RELEASE},app.kubernetes.io/component=temporal" \
    --timeout=120s

  if [[ "${MONITORING_ENABLED}" == "true" ]]; then
    kubectl -n "${NAMESPACE}" rollout status deployment \
      -l "app.kubernetes.io/instance=${RELEASE},app.kubernetes.io/component=prometheus" \
      --timeout=120s
    kubectl -n "${NAMESPACE}" rollout status deployment \
      -l "app.kubernetes.io/instance=${RELEASE},app.kubernetes.io/component=grafana" \
      --timeout=120s
  fi

  wait_for_api_health

  if [[ "${MONITORING_ENABLED}" == "true" ]]; then
    echo
    echo "Grafana health:"
    curl -fsS "http://${GRAFANA_HOST}/api/health" || true
    echo
  fi
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
