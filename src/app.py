# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from factory_graph import PUBLISH_STATUS, nim_factory_graph_pipeline

app = FastAPI(title="NIM Automation Factory Control Plane")

# System Observability Telemetry Setup
FACTORY_PIPELINE_RUNS = Counter(
    "nim_factory_pipeline_runs_total",
    "Total number of executions handled by the NIM model-to-service factory.",
    ["model_name", "target_gpu", "status"]
)
INFERENCE_THROUGHPUT_GAUGE = Histogram(
    "nim_factory_validated_throughput_tps",
    "Tokens per second metrics captured during automated validation phases.",
    ["model_name", "target_gpu", "status"],
    buckets=[500, 1000, 1500, 2000, 2500, 3000, 3500]
)

class ModelIngestRequest(BaseModel):
    model_name: str
    target_gpu: str  # e.g., "NVIDIA-GB200", "NVIDIA-B300", "AMD-MI355X"

@app.get("/metrics")
def get_system_metrics():
    """Exposes Prometheus-native metrics endpoints for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health_check():
    """Returns a lightweight readiness signal for local orchestration."""
    return {"status": "ok"}


@app.post("/factory/ingest")
def trigger_factory_pipeline(request: ModelIngestRequest):
    """Triggers the automated compilation and validation graph for a raw AI model."""
    initial_state = {
        "model_name": request.model_name,
        "target_gpu": request.target_gpu,
        "hardware_topology": {},
        "validation_results": {},
        "status": "Ingested",
        "error_message": ""
    }
    
    try:
        # Execute the LangGraph workflow synchronously
        final_output = nim_factory_graph_pipeline.invoke(initial_state)
        
        # Record pipeline metrics to Prometheus telemetry
        status_outcome = final_output.get("status", "Unknown")
        FACTORY_PIPELINE_RUNS.labels(
            model_name=request.model_name, 
            target_gpu=request.target_gpu, 
            status=status_outcome
        ).inc()
        
        tps_metric = final_output.get("validation_results", {}).get("metrics", {}).get("tokens_per_second", 0)
        if tps_metric > 0:
            INFERENCE_THROUGHPUT_GAUGE.labels(
                model_name=request.model_name,
                target_gpu=request.target_gpu,
                status=status_outcome
            ).observe(tps_metric)
        return {
            "message": "Factory workflow iteration finalized.",
            "pipeline_summary": {
                "model": final_output["model_name"],
                "hardware": final_output["target_gpu"],
                "topology_used": final_output["hardware_topology"],
                "test_metrics": final_output["validation_results"].get("metrics", {}),
                "final_status": status_outcome,
                "deployable": status_outcome == PUBLISH_STATUS,
                "error_log": final_output["error_message"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Factory Pipeline failure: {str(e)}")

if __name__ == "__main__":
    print("Launching Local NIM Factory Control Plane on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
