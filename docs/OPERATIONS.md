# Local Production Runbook

This runbook treats the local laptop as the production-like environment for the
Model Factory POC.

## Docker Compose Path

Start:

```bash
./scripts/local-docker-compose.sh
```

The start script writes `.runtime.env` with:

```bash
BASE_URL=http://localhost:8000
```

Health:

```bash
source .runtime.env
curl -sS "${BASE_URL}/health" | jq .
```

Smoke test:

```bash
./scripts/smoke-test.sh
```

Load test:

```bash
REQUESTS=30 CONCURRENCY=6 ./scripts/load-test.sh
```

Stop:

```bash
./scripts/uninstall-docker-compose.sh
```

## Local Kubernetes Helm Path

Start:

```bash
./scripts/local-helm-deployment.sh
```

The start script writes `.runtime.env` with the ingress URL:

```bash
BASE_URL=http://model-factory.localhost
```

Health:

```bash
source .runtime.env
curl -sS "${BASE_URL}/health" | jq .
```

Smoke test:

```bash
./scripts/smoke-test.sh
```

Load test:

```bash
REQUESTS=30 CONCURRENCY=6 ./scripts/load-test.sh
```

Stop release:

```bash
./scripts/uninstall-helm-deployment.sh
```

Stop release and delete namespace:

```bash
DELETE_NAMESPACE=true ./scripts/uninstall-helm-deployment.sh
```

## Validation Gate

Run this before treating local changes as healthy:

```bash
./scripts/validate-local.sh
```

The validation gate runs unit tests, checks shell script syntax, and renders the
Helm chart with ingress enabled.
