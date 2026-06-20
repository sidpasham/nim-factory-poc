#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

cat > .runtime.env <<'EOF'
RUNTIME_MODE=docker-compose
BASE_URL=http://localhost:8000
GRAFANA_URL=http://localhost:3000
PROMETHEUS_URL=http://localhost:9090
TEMPORAL_UI_URL=http://localhost:8233
EOF
echo "Wrote .runtime.env for Docker Compose"

docker compose --env-file .env -f docker-compose.yml up --build
