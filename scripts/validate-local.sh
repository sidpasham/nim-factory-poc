#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

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

echo "Running unit tests"
PYTHONPATH=src "${PYTHON_BIN}" -m unittest discover -s test

echo "Checking shell script syntax"
for script in scripts/*.sh; do
  bash -n "${script}"
done

echo "Checking markdown bash examples"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

for markdown_path in [Path("examples/benchmark-curl-examples.md")]:
    blocks = []
    in_block = False
    current = []
    for line in markdown_path.read_text().splitlines():
        if line.strip() == "```bash":
            in_block = True
            current = []
            continue
        if in_block and line.strip() == "```":
            blocks.append("\n".join(current))
            in_block = False
            continue
        if in_block:
            current.append(line)

    target = Path("/tmp") / f"{markdown_path.stem}.sh"
    target.write_text("\n\n".join(blocks) + "\n")
    print(f"wrote {len(blocks)} bash blocks from {markdown_path} to {target}")
PY
bash -n /tmp/benchmark-curl-examples.sh

echo "Rendering Helm chart"
helm template llm-gpu-benchmarking helm/llm-gpu-benchmarking \
  --namespace llm-gpu-benchmarking \
  --set ingress.enabled=true \
  --set ingress.className=traefik \
  --set "ingress.hosts[0].host=llm-gpu-benchmarking.localhost" \
  --set "ingress.hosts[0].paths[0].path=/" \
  --set "ingress.hosts[0].paths[0].pathType=Prefix" >/tmp/llm-gpu-benchmarking-rendered.yaml

echo "Validation completed"
