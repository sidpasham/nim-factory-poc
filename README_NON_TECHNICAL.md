# Model Factory POC - Non-Technical Overview

## Executive Summary

This proof of concept shows how an AI platform could decide whether a large
language model is ready to run on a selected hardware target.

A user submits a model, a hardware choice, a deployment environment, and a
precision mode. The system checks whether the model is likely to fit, runs a
simulated benchmark, records the result, and shows operational signals in a
dashboard.

The important change is that the demo no longer treats every request as a
success. If someone tries to place a very large model on hardware with too
little memory, the system fails the request for a specific reason:

```text
VRAM_INSUFFICIENT_EXCEPTION
```

That makes the demo closer to how a real production factory would behave.

## What This System Does

In plain language, the system acts like a factory intake desk for AI models:

1. A user asks to prepare a model for a target environment.
2. The system checks the selected hardware profile.
3. The system estimates how much GPU memory the model needs.
4. The system adjusts that estimate based on precision mode.
5. The system decides whether the model can fit.
6. The system records success, failure, speed, latency, and resource use.
7. The dashboard shows what happened across many requests.

## Why This Matters

Large AI models are expensive to run. Before a team spends time compiling,
deploying, or testing one, the platform should catch obvious failures early.

Example:

- A 70B model in `FP16` needs far more memory than a single 24GB A10G card has.
- The system should reject that combination before pretending it can deploy.
- The failure should be visible in logs, API status, Prometheus metrics, and
  Grafana dashboards.

This is the behavior the validation matrix provides.

## Key Terms

| Term | Meaning |
| --- | --- |
| model factory | A control plane that prepares model-serving deployments. |
| Model | The AI model a user wants to serve, such as `Llama-3-70B`. |
| Target hardware | The GPU profile the user wants to run on, such as `H100-80GB` or `A10G-24GB`. |
| VRAM | GPU memory. Large models need enough VRAM to load and run. |
| Precision mode | The memory and speed profile selected for the model: `FP16`, `INT8`, or `INT4`. |
| Validation matrix | A deterministic compatibility check between model size, precision mode, and hardware capacity. |
| Temporal | The workflow engine that keeps long-running factory work separate from the API request. |
| Prometheus | The metrics collector. |
| Grafana | The dashboard used to view metrics. |

## What The User Can Choose

Every request can include:

| Choice | Example | Why it matters |
| --- | --- | --- |
| Model | `Llama-3-70B` | Bigger models need more GPU memory. |
| Hardware | `A10G-24GB` | Smaller GPUs have less memory and lower expected throughput. |
| Environment | `kubernetes` | Labels the target platform for routing and reporting. |
| Precision mode | `FP16`, `INT8`, or `INT4` | Lower precision usually needs less memory and can run faster, with a simulated quality tradeoff. |

## Example Outcomes

### Expected Failure

Request:

```text
Model: Llama-3-70B
Hardware: A10G-24GB
Precision mode: FP16
```

Outcome:

```text
Failed
Reason: VRAM_INSUFFICIENT_EXCEPTION
```

Why:

The model needs about 158.8GB of GPU memory in this demo's estimate, but the
target card has 24GB.

### Expected Success

Request:

```text
Model: Llama-3-70B
Hardware: H100-80GB
Precision mode: INT4
```

Outcome:

```text
Ready to deploy in the demo pipeline
```

Why:

The lower precision mode reduces the estimated memory footprint enough to fit on
the H100 profile.

## What The Dashboard Shows

The Grafana dashboard is designed to tell an operational story:

- How many factory requests were submitted.
- How many succeeded and failed.
- Which models are being tested.
- Which hardware profiles are being used.
- Which precision modes are producing better throughput.
- Whether failures are coming from memory limits or benchmark thresholds.
- How long each pipeline stage takes.
- How many worker jobs are active.

This helps a viewer understand whether the factory is healthy, whether workers
are busy, and where bottlenecks appear.

## How The Pieces Fit Together

| Piece | Role |
| --- | --- |
| FastAPI | Receives user requests and returns workflow IDs. |
| Temporal | Tracks the long-running workflow. |
| Worker | Executes the model validation pipeline away from the API process. |
| LangGraph | Defines the factory stages and routing logic. |
| Validation matrix | Makes the fit/fail decision based on model size, hardware memory, and precision mode. |
| Prometheus | Collects metrics from the API and worker. |
| Grafana | Shows dashboards from those metrics. |

## What Is Real And What Is Simulated

Real in this POC:

- API request handling.
- Background workflow execution.
- Worker process separation.
- Config-backed hardware and deployment profiles.
- Deterministic validation decisions.
- Metrics and dashboards.
- Docker Compose stack.
- Local Kubernetes Helm run.

Simulated in this POC:

- Real GPU provisioning.
- Real model compilation.
- Real artifact publishing.
- Real hardware benchmark execution in the default mode.

Optional:

- The project can call a hosted NVIDIA OpenAI-compatible endpoint when
  `MODEL_FACTORY_VALIDATION_MODE=hosted` and an API key is configured.

## How To Demo It

Start the local stack:

```bash
./scripts/local-docker-compose.sh
```

Submit a few requests:

```bash
REQUESTS=30 CONCURRENCY=6 ./scripts/load-test.sh
```

Open:

| Tool | URL |
| --- | --- |
| API | http://localhost:8000 |
| Temporal UI | http://localhost:8233 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Grafana uses the local demo login:

```text
admin / admin
```

## Production Readiness View

This POC has the right high-level shape for a production discussion: API,
workflow engine, worker process, validation rules, metrics, and dashboards.

It is not yet production ready. A production version would still need:

- Real GPU inventory and scheduling.
- Real model-service build and deployment steps.
- A production Temporal cluster.
- Authentication and authorization.
- Durable storage for run history and artifacts.
- Centralized logs and tracing.
- Alerts for failures and slow stages.
- Strong secrets management.
- CI/CD, dependency locking, image scanning, and Kubernetes hardening.

The value of this project is that it makes those requirements concrete and easy
to discuss.
