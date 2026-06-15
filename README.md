# NIM Factory POC

A local proof of concept for a model-to-service factory control plane.

The app accepts an LLM model name, target hardware, and target environment,
looks up config-backed topology and deployment metadata, validates the model by
calling NVIDIA's hosted NIM-compatible API, routes the result through a
LangGraph workflow, and exports Prometheus metrics for a Grafana dashboard.

This is a POC/demo project, not production infrastructure. It is designed to
show the factory control-plane flow with real NVIDIA hosted model calls while
running locally through Docker Compose or Rancher Desktop Kubernetes.

## What This Demonstrates

- FastAPI control-plane endpoint for model ingestion.
- LangGraph workflow orchestration with explicit success/failure routing.
- Config-backed hardware profiles instead of hardcoded mock data.
- Config-backed deployment target profiles for cloud, on-prem, and Kubernetes.
- Real NIM validation using NVIDIA hosted OpenAI-compatible endpoints.
- Prometheus counters and histograms labeled by model, hardware, environment, and status.
- Docker Compose stack with Prometheus and a provisioned Grafana dashboard.
- Helm chart for Rancher Desktop or another Kubernetes cluster.
- Unit tests for routing, failure handling, and provisioning behavior.

## Architecture

```text
Client
  |
  | POST /factory/ingest
  v
FastAPI control plane
  |
  v
LangGraph workflow
  |
  +-- discover_infrastructure
  |     reads src/config/hardware_profiles.json
  |     reads src/config/deployment_targets.json
  |
  +-- run_validation_harness
  |     calls /v1/health/ready, /v1/models, /v1/chat/completions
  |
  +-- route_validation_results
        |
        +-- compile_and_publish_nim   -> NIM_Ready_To_Deploy
        +-- handle_failure            -> Failed

Prometheus scrapes /metrics
Grafana reads Prometheus and renders infra/grafana/dashboards/dashboard.json
```

## Project Layout

```text
src/                              Python application source
src/config/                       Hardware, deployment target, and model payload config
src/nim_runtime_client.py         NVIDIA hosted NIM-compatible client
test/                             Unit tests
examples/                         Copy/paste curl examples for demo sweeps
infra/                            Local infrastructure files
infra/Dockerfile                  App container image
infra/docker-compose.yml          App + Prometheus + Grafana stack
infra/helm/nim-factory/           Helm chart for Kubernetes deployment
infra/prometheus/prometheus.yml   Prometheus scrape config
infra/grafana/dashboards/         Provisioned Grafana dashboard JSON
infra/grafana/provisioning/       Grafana datasource and dashboard providers
```

## Prerequisites

- Python 3.12+
- Rancher Desktop with Docker-compatible CLI or `nerdctl`
- Docker Compose for the local observability stack
- `kubectl` and `helm` for the Kubernetes demo
- NVIDIA API key for hosted NIM calls

## Run Locally With Docker Compose

Copy the example environment file and add your NVIDIA API key:

```bash
cp .env.example .env
```

The local `.env` file is ignored by git. Do not commit it.

Start the local API, Prometheus, and Grafana stack:

```bash
docker compose --env-file .env -f infra/docker-compose.yml up --build
```

Services:

| Service | URL | Notes |
| --- | --- | --- |
| API | http://localhost:8000 | FastAPI app |
| Health | http://localhost:8000/health | Lightweight health check |
| Metrics | http://localhost:8000/metrics | Prometheus metrics |
| Prometheus | http://localhost:9090 | Local metrics store |
| Grafana | http://localhost:3000 | Login with `admin` / `admin` |

The Grafana dashboard is provisioned into the `NIM Factory` folder. The source
dashboard JSON lives at:

```text
infra/grafana/dashboards/dashboard.json
```

## Send Sample Requests

Run a few ingests so Prometheus and Grafana have data:

```bash
curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA GB300","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"z-ai/glm-5.1","target_gpu":"NVIDIA B200","target_environment":"cloud"}' \
  | jq .
```

More model, hardware, and environment variations are in:

```text
examples/factory-ingest-curl-examples.md
```

## NVIDIA Hosted NIM Runtime

The app always validates through the configured NIM-compatible endpoint. For the
hosted NVIDIA API, `.env` should look like:

```bash
NIM_MINIMUM_TPS_THRESHOLD=100
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_API_KEY=your-nvidia-api-key
NIM_TIMEOUT_SECONDS=30
```

The validation node:

1. Calls `GET /v1/health/ready`.
2. Calls `GET /v1/models` and verifies the requested `model_name`.
3. Calls `POST /v1/chat/completions` with a short benchmark prompt.
4. Computes TPS from completion tokens and elapsed latency.

Hosted OpenAI-compatible endpoints may not expose every self-hosted management
endpoint. If `/v1/health/ready` is unavailable, the client records the health
check as unsupported and continues with model listing and chat completion.

NVIDIA samples often use model-specific `max_tokens`, streaming, and reasoning
options. The validation profiles keep those request differences in config, and
some slower profiles use lower token limits so repeated runs stay practical for
demos. Model-specific request payloads live in:

```text
src/config/model_profiles.json
```

The project currently includes payload profiles for:

- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- `minimaxai/minimax-m3`
- `deepseek-ai/deepseek-v4-flash`
- `z-ai/glm-5.1`
- `meta/llama-3.1-8b-instruct`
- `minimaxai/minimax-m2.7`

The code does not store, print, or export `NVIDIA_API_KEY`.

## Kubernetes Demo With Rancher Desktop

Build the local app image:

```bash
docker compose --env-file .env -f infra/docker-compose.yml build app
```

Create the Kubernetes namespace and secret:

```bash
kubectl create namespace nim-factory --dry-run=client -o yaml | kubectl apply -f -
kubectl -n nim-factory create secret generic nim-factory-secrets \
  --from-literal=NVIDIA_API_KEY="$NVIDIA_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -
```

Install with Helm:

```bash
helm upgrade --install nim-factory infra/helm/nim-factory \
  --namespace nim-factory
```

Port-forward the service:

```bash
kubectl -n nim-factory port-forward svc/nim-factory 8001:8000
```

Call the Kubernetes-hosted factory API:

```bash
curl -sS -X POST http://localhost:8001/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .
```

This Kubernetes demo runs the factory control plane as a real Kubernetes
workload. The model endpoint remains NVIDIA hosted NIM; the POC is not claiming
to measure self-hosted GPU node performance.

Example response:

```json
{
  "message": "Factory workflow iteration finalized.",
  "pipeline_summary": {
    "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "hardware": "NVIDIA GB200",
    "environment": "kubernetes",
    "topology_used": {
      "accelerator": "NVIDIA GB200",
      "interconnect": "NVLink",
      "pcie_generation": "Gen5",
      "smartnic_active": true,
      "optimized_driver": "nvidia_datacenter_blackwell"
    },
    "deployment_target": {
      "environment": "kubernetes",
      "platform": "Rancher Desktop k3s",
      "control_plane": "local Kubernetes",
      "nim_endpoint_mode": "nvidia_hosted_external"
    },
    "test_metrics": {
      "tokens_per_second": 145.97,
      "error_rate": 0.0,
      "latency_ms": 657.66,
      "output_tokens": 96
    },
    "final_status": "NIM_Ready_To_Deploy",
    "deployable": true,
    "error_log": ""
  }
}
```

## Dashboard

The Grafana dashboard focuses on these POC questions:

- How many total, successful, and failed factory runs happened in the selected time range?
- What is the success rate?
- What are average and p95 TPS?
- Which hardware targets, environments, and LLM models are being exercised?
- What TPS is each model/hardware pair producing over time?
- What average throughput and latency is each LLM model producing?

Dashboard panels include:

- `Total Runs`
- `Success Runs`
- `Failure Runs`
- `Success Rate`
- `Average TPS`
- `P95 TPS`
- `Runs by LLM Model`
- `Hardware by Status`
- `Average TPS by Hardware`
- `Average Throughput by LLM Model`
- `Average Latency by LLM Model`
- `TPS by LLM Model and Hardware Over Time`
- `Average TPS by Model, Hardware, and Environment`
- `Runs by Model, Hardware, Environment, and Status`

Dashboard filters:

- `LLM Model`
- `Hardware`
- `Environment`

Filter values are preconfigured from the demo model, hardware, and environment
profiles so the dashboard is usable before every option has generated metrics.

Grafana can edit provisioned dashboards in the UI, but UI edits are stored in
Grafana's database. To keep dashboard changes in source control, export or copy
the final JSON back into `infra/grafana/dashboards/dashboard.json`.

## Prometheus Metrics

The app exports:

| Metric | Type | Labels | Purpose |
| --- | --- | --- | --- |
| `nim_factory_pipeline_runs_total` | Counter | `model_name`, `target_gpu`, `target_environment`, `status` | Counts factory runs by outcome |
| `nim_factory_validated_throughput_tps` | Histogram | `model_name`, `target_gpu`, `target_environment`, `status` | Records real NIM validation TPS values |
| `nim_factory_validated_latency_ms` | Histogram | `model_name`, `target_gpu`, `target_environment`, `status` | Records real NIM validation latency values |

## Configuration

Hardware matching lives in:

```text
src/config/hardware_profiles.json
```

The configured demo hardware targets are:

- `NVIDIA GB200`
- `NVIDIA GB300`
- `NVIDIA B200`
- `AMD MI355X`
- `AMD MI455X`

Deployment target matching lives in:

```text
src/config/deployment_targets.json
```

Model request payloads live in:

```text
src/config/model_profiles.json
```

To add a new hardware profile, add a `match_any` token and topology block in
`hardware_profiles.json`. To add a new environment target, add a `match_any`
token and target block in `deployment_targets.json`. To add a new NVIDIA hosted
model, add an exact model profile in `model_profiles.json`. Unknown hardware
and environment labels are rejected instead of mapped to a generic fallback.

Runtime validation is controlled by environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NIM_MINIMUM_TPS_THRESHOLD` | `100` | Minimum TPS required for the workflow to publish |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Base URL for the NIM-compatible endpoint |
| `NVIDIA_API_KEY` | empty | Bearer token for NVIDIA hosted NIM |
| `NIM_TIMEOUT_SECONDS` | `30` | HTTP timeout for remote NIM calls |
| `NIM_MODEL_PROFILES_PATH` | `src/config/model_profiles.json` | Optional override for model payload profiles |

## Run Without Docker

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the API:

```bash
PYTHONPATH=src uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s test
```

## Test Suite

The tests cover:

- Hardware profile selection from JSON config.
- Deployment target profile selection from JSON config.
- Remote NIM client request/response handling.
- Real NIM validation delegation.
- Success/failure routing.
- Contextual failure message construction.

Run:

```bash
python -m unittest discover -s test
```

## Public Repo Notes

- NVIDIA hosted NIM integration requires user-provided API access and is subject
  to NVIDIA and model-provider terms.
- The Grafana password is the local demo default `admin` / `admin`; do not use it
  for a shared or production deployment.
- The POC measures a small request against a configured NVIDIA hosted NIM
  endpoint. It is a lightweight validation check, not a production load test.
- Product names such as NVIDIA, NIM, and GPU model names are used only to make
  the POC concrete. This project is not affiliated with or endorsed by
  those vendors.

## Limitations And Next Steps

- Add a real inventory/provisioning API for hardware topology discovery.
- Add OKE/on-prem GPU node profiles when validating self-hosted NIM runtimes.
- Optionally scrape NIM's native `/v1/metrics` endpoint and NVIDIA DCGM exporter
  for runtime and GPU telemetry.
- Persist workflow runs if historical replay is needed outside Prometheus.
- Add structured logging and request IDs.
- Add CI to run unit tests and JSON validation on pull requests.
- Pin or lock dependencies before using this outside a local POC.
