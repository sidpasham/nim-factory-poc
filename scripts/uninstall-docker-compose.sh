#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  docker compose --env-file .env -f docker-compose.yml down --volumes --remove-orphans
else
  docker compose -f docker-compose.yml down --volumes --remove-orphans
fi

if [[ -f ".runtime.env" ]] && grep -q '^RUNTIME_MODE=docker-compose$' .runtime.env; then
  rm .runtime.env
  echo "Removed Docker Compose .runtime.env"
fi
