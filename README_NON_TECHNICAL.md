# NIM Factory POC - Non-Technical Overview

## Executive Summary

This proof of concept shows how an AI model can move through a simple factory
pipeline before it is considered ready to deploy. The pipeline accepts a model,
checks the target hardware and environment, validates the model through real
NVIDIA hosted NIM API calls, and shows the results in a Grafana dashboard.

This is a demo and learning project, not a production system. Its purpose is to
make the NIM factory concept easier to understand and discuss.

## What Value This POC Gives

- Shows the end-to-end idea of taking an AI model and validating whether it is
  ready for deployment.
- Demonstrates how the same factory workflow can track cloud, on-prem, and
  Kubernetes deployment targets.
- Uses real NVIDIA hosted model calls instead of fake model responses.
- Provides a dashboard that shows total runs, successful runs, failed runs,
  throughput, latency, model comparisons, hardware labels, and environment
  labels.
- Helps teams discuss what data, interfaces, checks, and observability would be
  needed for a larger NIM factory platform.

## What Technologies Are Used and Why

- **NVIDIA hosted NIM API**: Used to make real model validation calls through
  NVIDIA's OpenAI-compatible endpoint.
- **FastAPI**: Provides a lightweight API where users can submit a model,
  hardware target, and deployment environment.
- **LangGraph**: Models the factory workflow as clear steps, such as discovery,
  validation, success routing, and failure handling.
- **Prometheus**: Collects metrics from the factory pipeline, including run
  counts, throughput, and latency.
- **Grafana**: Displays the metrics in a dashboard that is easy to review during
  demos.
- **Docker Compose**: Runs the local API, Prometheus, and Grafana stack together
  for quick demos.
- **Helm and Kubernetes**: Show how the factory control plane can also run as a
  Kubernetes workload, such as with Rancher Desktop on a laptop.

