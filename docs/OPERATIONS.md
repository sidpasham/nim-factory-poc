# Local Production Runbook

This runbook treats the local laptop as the production-like environment for
LLM GPU Benchmarking.

## Local Kubernetes Path

Start:

```bash
./scripts/local-helm-deployment.sh up
```

The start script writes `.runtime.env` with the ingress URL:

```bash
BASE_URL=http://llm-gpu-benchmarking.localhost
```

Health:

```bash
source .runtime.env
curl -sS "${BASE_URL}/health" | jq .
```

End-to-end test:

```bash
./scripts/local-helm-deployment.sh test
```

Load test:

```bash
REQUESTS=30 CONCURRENCY=6 ./scripts/local-helm-deployment.sh load
```

Stop release:

```bash
./scripts/local-helm-deployment.sh down
```

Stop release and delete namespace:

```bash
DELETE_NAMESPACE=true ./scripts/local-helm-deployment.sh down
```

Status:

```bash
./scripts/local-helm-deployment.sh status
```

## Validation Gate

Run this before treating local changes as healthy:

```bash
./scripts/validate-local.sh
```

The validation gate runs unit tests, checks shell script syntax, and renders the
Helm chart with ingress enabled.
