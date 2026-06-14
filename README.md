# NIM Factory POC

A local proof of concept for a model-to-service factory control plane.

The app accepts an LLM model name and target GPU, looks up a simulated hardware
topology from JSON config, runs a simulated validation harness, routes the result
through a LangGraph workflow, and exports Prometheus metrics for a Grafana
dashboard.

This is a simulation-oriented POC. It does not build real NVIDIA NIM containers,
provision real hardware, or call external infrastructure APIs.

## What This Demonstrates

- FastAPI control-plane endpoint for model ingestion.
- LangGraph workflow orchestration with explicit success/failure routing.
- Config-backed hardware profiles instead of hardcoded mock data.
- Prometheus counters and histograms labeled by model, hardware, and status.
- Docker Compose stack with Prometheus and a provisioned Grafana dashboard.
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
  |
  +-- run_validation_harness
  |     reads src/config/validation_profiles.json
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
src/config/                       Hardware and validation simulation config
test/                             Unit tests
infra/                            Local infrastructure files
infra/Dockerfile                  App container image
infra/docker-compose.yml          App + Prometheus + Grafana stack
infra/prometheus/prometheus.yml   Prometheus scrape config
infra/grafana/dashboards/         Provisioned Grafana dashboard JSON
infra/grafana/provisioning/       Grafana datasource and dashboard providers
```

## Prerequisites

- Python 3.12+
- Docker and Docker Compose

## Run Locally With Docker Compose

```bash
docker compose -f infra/docker-compose.yml up --build
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
curl -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"llama-3.1-8b","target_gpu":"NVIDIA-GB200"}'

curl -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"mistral-7b","target_gpu":"AMD-MI355X"}'

curl -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"mixtral-8x7b","target_gpu":"GENERIC-GPU"}'
```

Example response:

```json
{
  "message": "Factory workflow iteration finalized.",
  "pipeline_summary": {
    "model": "llama-3.1-8b",
    "hardware": "NVIDIA-GB200",
    "topology_used": {
      "interconnect": "NVLink",
      "pcie_generation": "Gen5",
      "smartnic_active": true,
      "optimized_driver": "v550_prod"
    },
    "test_metrics": {
      "tokens_per_second": 2700,
      "error_rate": 0.0,
      "latency_ms": 18.5
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
- Which hardware targets and LLM models are being exercised?
- What TPS is each model/hardware pair producing over time?

Dashboard panels include:

- `Total Runs`
- `Success Runs`
- `Failure Runs`
- `Success Rate`
- `Average TPS`
- `P95 TPS`
- `Runs by Hardware`
- `Runs by LLM Model`
- `Hardware by Status`
- `TPS by LLM Model and Hardware Over Time`
- `Average TPS by Model and Hardware`
- `Runs by Model, Hardware, and Status`

Dashboard filters:

- `LLM Model`
- `Hardware`

Grafana can edit provisioned dashboards in the UI, but UI edits are stored in
Grafana's database. To keep dashboard changes in source control, export or copy
the final JSON back into `infra/grafana/dashboards/dashboard.json`.

## Prometheus Metrics

The app exports:

| Metric | Type | Labels | Purpose |
| --- | --- | --- | --- |
| `nim_factory_pipeline_runs_total` | Counter | `model_name`, `target_gpu`, `status` | Counts factory runs by outcome |
| `nim_factory_validated_throughput_tps` | Histogram | `model_name`, `target_gpu`, `status` | Records simulated TPS values |

## Configuration

Hardware matching lives in:

```text
src/config/hardware_profiles.json
```

Validation harness behavior lives in:

```text
src/config/validation_profiles.json
```

To add a new hardware profile, add a `match_any` token and topology block in
`hardware_profiles.json`, then add or reuse an interconnect profile in
`validation_profiles.json`.

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
- Validation harness metric generation.
- Success/failure routing.
- Contextual failure message construction.

Run:

```bash
python -m unittest discover -s test
```

## Public Repo Notes

- No cloud credentials, API keys, or secrets are required.
- The Grafana password is the local demo default `admin` / `admin`; do not use it
  for a shared or production deployment.
- The app uses simulated random validation outcomes. Dashboard values are useful
  for demonstrating observability wiring, not for real performance claims.
- Product names such as NVIDIA, NIM, and GPU model names are used only to make
  the simulation concrete. This project is not affiliated with or endorsed by
  those vendors.

## Limitations And Next Steps

- Replace the simulated provisioning adapter with a real inventory/provisioning API.
- Persist workflow runs if historical replay is needed outside Prometheus.
- Add structured logging and request IDs.
- Add CI to run unit tests and JSON validation on pull requests.
- Pin or lock dependencies before using this outside a local POC.
