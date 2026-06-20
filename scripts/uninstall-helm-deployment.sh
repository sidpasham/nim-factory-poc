#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NAMESPACE="${NAMESPACE:-model-factory}"
RELEASE="${RELEASE:-model-factory}"
DELETE_NAMESPACE="${DELETE_NAMESPACE:-false}"

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
