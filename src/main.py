import asyncio
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from temporalio.client import Client

from metrics import PIPELINE_DURATION_SECONDS
from schemas import ModelIngestRequest
from workflows import ModelFactoryWorkflow


temporal_client: Optional[Client] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global temporal_client
    temporal_client = await _connect_temporal()
    try:
        yield
    finally:
        temporal_client = None


app = FastAPI(
    title="Model Factory Control Plane",
    lifespan=lifespan,
)


def _workflow_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")[:80] or "model"


async def _connect_temporal() -> Client:
    temporal_address = os.getenv("TEMPORAL_ADDRESS", "temporal:7233")
    attempts = int(os.getenv("TEMPORAL_CONNECT_ATTEMPTS", "3"))
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await Client.connect(temporal_address)
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(min(attempt, 5))

    raise RuntimeError(
        f"failed to connect to Temporal at {temporal_address} after {attempts} attempts"
    ) from last_error


@app.get("/metrics")
def get_system_metrics():
    """Exposes API-process Prometheus metrics for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health_check():
    """Returns a lightweight readiness signal for local orchestration."""
    return {
        "status": "ok",
        "temporal_connected": temporal_client is not None,
    }


@app.post("/factory/ingest")
async def trigger_factory_pipeline(request: ModelIngestRequest):
    """Starts the distributed factory workflow asynchronously through Temporal."""
    if temporal_client is None:
        raise HTTPException(status_code=503, detail="Temporal client is not connected")

    start_time = time.perf_counter()
    precision_mode = request.precision_mode.value
    workflow_id = (
        f"model-factory-{_workflow_slug(request.model_name)}-"
        f"{precision_mode.lower()}-{uuid.uuid4().hex[:8]}"
    )

    initial_state = {
        "model_name": request.model_name,
        "target_gpu": request.target_gpu,
        "target_environment": request.target_environment,
        "precision_mode": precision_mode,
        "hardware_topology": {},
        "deployment_target": {},
        "precision_result": {},
        "compile_result": {},
        "validation_results": {},
        "stage_durations": {},
        "status": "Ingested",
        "error_message": "",
    }

    try:
        await temporal_client.start_workflow(
            ModelFactoryWorkflow.run,
            initial_state,
            id=workflow_id,
            task_queue="model-factory-task-queue",
        )
        duration_seconds = time.perf_counter() - start_time
        PIPELINE_DURATION_SECONDS.labels(
            stage="ingest",
            model_name=request.model_name,
            target_gpu=request.target_gpu,
            target_environment=request.target_environment,
            precision_mode=precision_mode,
            status="STARTED",
        ).observe(duration_seconds)

        return {
            "message": "Distributed Model Factory compilation pipeline initiated.",
            "workflow_id": workflow_id,
            "status_url": f"/factory/status/{workflow_id}",
            "precision_mode": precision_mode,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start factory pipeline: {exc}",
        ) from exc


@app.get("/factory/status/{workflow_id}")
async def get_factory_status(workflow_id: str):
    """Queries live status and final result for a factory pipeline."""
    if temporal_client is None:
        raise HTTPException(status_code=503, detail="Temporal client is not connected")

    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        description = await handle.describe()

        response_data = {
            "workflow_id": workflow_id,
            "status": description.status.name,
        }

        if description.status.name == "COMPLETED":
            response_data["pipeline_summary"] = await handle.result()

        return response_data
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow track not found: {exc}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
