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
REQUESTS="${REQUESTS:-30}"
CONCURRENCY="${CONCURRENCY:-6}"
POLL="${POLL:-true}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=.venv/bin/python
  elif [[ -x "venv/bin/python" ]]; then
    PYTHON_BIN=venv/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi

echo "Using BASE_URL=${BASE_URL}"
"${PYTHON_BIN}" examples/load_test_validation_matrix.py \
  --base-url "${BASE_URL}" \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --poll "${POLL}"
