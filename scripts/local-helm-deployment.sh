#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NAMESPACE="${NAMESPACE:-model-factory}"
RELEASE="${RELEASE:-model-factory}"
INGRESS_CLASS="${INGRESS_CLASS:-traefik}"
INGRESS_HOST="${INGRESS_HOST:-model-factory.localhost}"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

set -a
source .env
set +a

kubectl cluster-info
kubectl get nodes

docker compose --env-file .env -f docker-compose.yml build app

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

if [[ -n "${NVIDIA_API_KEY:-}" ]]; then
  kubectl -n "${NAMESPACE}" create secret generic model-factory-secrets \
    --from-literal=NVIDIA_API_KEY="${NVIDIA_API_KEY}" \
    --dry-run=client -o yaml | kubectl apply -f -
else
  echo "NVIDIA_API_KEY is not set; skipping optional hosted-validation secret."
fi

helm upgrade --install "${RELEASE}" helm/model-factory \
  --namespace "${NAMESPACE}" \
  --set ingress.enabled=true \
  --set "ingress.className=${INGRESS_CLASS}" \
  --set "ingress.hosts[0].host=${INGRESS_HOST}" \
  --set "ingress.hosts[0].paths[0].path=/" \
  --set "ingress.hosts[0].paths[0].pathType=Prefix" \
  --set-string "config.minimumTpsThreshold=${MODEL_FACTORY_MINIMUM_TPS_THRESHOLD:-100}" \
  --set-string "config.nimBaseUrl=${NIM_BASE_URL:-https://integrate.api.nvidia.com/v1}" \
  --set-string "config.nimTimeoutSeconds=${NIM_TIMEOUT_SECONDS:-30}" \
  --set-string "config.validationMode=${MODEL_FACTORY_VALIDATION_MODE:-validation}" \
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

echo
echo "API health:"
for attempt in {1..20}; do
  if curl -fsS "http://${INGRESS_HOST}/health"; then
    echo
    cat > .runtime.env <<EOF
RUNTIME_MODE=kubernetes-helm
BASE_URL=http://${INGRESS_HOST}
KUBERNETES_NAMESPACE=${NAMESPACE}
HELM_RELEASE=${RELEASE}
INGRESS_CLASS=${INGRESS_CLASS}
INGRESS_HOST=${INGRESS_HOST}
EOF
    echo "Wrote .runtime.env for local Kubernetes Helm"
    exit 0
  fi
  sleep 2
  echo "Waiting for ingress health check (${attempt}/20)..."
done

echo "API health check failed at http://${INGRESS_HOST}/health" >&2
exit 1
