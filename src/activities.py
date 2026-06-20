import logging
from typing import Any, Dict

from temporalio import activity

from factory_graph import PUBLISH_STATUS, model_factory_graph_pipeline
from metrics import (
    ACTIVE_WORKERS,
    FACTORY_FAILURES,
    FACTORY_PIPELINE_RUNS,
    INFERENCE_LATENCY_GAUGE,
    INFERENCE_THROUGHPUT_GAUGE,
    MODEL_ACCURACY_SCORE,
    PIPELINE_DURATION_SECONDS,
    VALIDATION_MATRIX_BENCHMARK_LATENCY_MS,
    VALIDATION_MATRIX_BENCHMARK_THROUGHPUT_TPS,
    VRAM_CAPACITY_GB,
    VRAM_REQUIRED_GB,
)


LOGGER = logging.getLogger(__name__)


def _failure_stage_and_reason(final_output: Dict[str, Any]) -> tuple[str, str]:
    compile_result = final_output.get("compile_result", {})
    validation_results = final_output.get("validation_results", {})
    return (
        compile_result.get("stage")
        or validation_results.get("stage")
        or "benchmark",
        compile_result.get("reason")
        or validation_results.get("reason")
        or validation_results.get("error_code")
        or "validation_failed",
    )


def _record_stage_durations(final_output: Dict[str, Any], status_outcome: str) -> None:
    precision_mode = final_output.get("precision_mode", "FP16")
    for stage, duration_seconds in final_output.get("stage_durations", {}).items():
        PIPELINE_DURATION_SECONDS.labels(
            stage=stage,
            model_name=final_output["model_name"],
            target_gpu=final_output["target_gpu"],
            target_environment=final_output["target_environment"],
            precision_mode=precision_mode,
            status=status_outcome,
        ).observe(duration_seconds)

    if (
        precision_mode == "FP16"
        and "precision_profile" not in final_output.get("stage_durations", {})
    ):
        PIPELINE_DURATION_SECONDS.labels(
            stage="precision_profile",
            model_name=final_output["model_name"],
            target_gpu=final_output["target_gpu"],
            target_environment=final_output["target_environment"],
            precision_mode=precision_mode,
            status="SKIPPED",
        ).observe(0.0)


def _record_validation_resource_metrics(final_output: Dict[str, Any]) -> None:
    compile_result = final_output.get("compile_result", {})
    validation_metrics = final_output.get("validation_results", {}).get("metrics", {})
    precision_mode = final_output.get("precision_mode", "FP16")

    required_vram_gb = compile_result.get("required_vram_gb") or validation_metrics.get(
        "required_vram_gb"
    )
    available_vram_gb = compile_result.get("available_vram_gb") or validation_metrics.get(
        "available_vram_gb"
    )
    accuracy_score = validation_metrics.get("accuracy_score")

    if required_vram_gb is not None:
        VRAM_REQUIRED_GB.labels(
            model_name=final_output["model_name"],
            target_gpu=final_output["target_gpu"],
            precision_mode=precision_mode,
        ).set(required_vram_gb)

    if available_vram_gb is not None:
        VRAM_CAPACITY_GB.labels(
            model_name=final_output["model_name"],
            target_gpu=final_output["target_gpu"],
            precision_mode=precision_mode,
        ).set(available_vram_gb)

    if accuracy_score is not None:
        MODEL_ACCURACY_SCORE.labels(
            model_name=final_output["model_name"],
            target_gpu=final_output["target_gpu"],
            precision_mode=precision_mode,
        ).set(accuracy_score)


@activity.defn
async def execute_compilation_and_validation(state: dict) -> dict:
    """Runs the staged LangGraph pipeline inside a Temporal worker activity."""
    ACTIVE_WORKERS.inc()
    try:
        activity.logger.info(
            "worker picked up payload model=%s hardware=%s environment=%s precision_mode=%s",
            state["model_name"],
            state["target_gpu"],
            state["target_environment"],
            state.get("precision_mode", "FP16"),
        )
        final_output = model_factory_graph_pipeline.invoke(state)
        status_outcome = final_output.get("status", "Unknown")

        FACTORY_PIPELINE_RUNS.labels(
            model_name=final_output["model_name"],
            target_gpu=final_output["target_gpu"],
            target_environment=final_output["target_environment"],
            status=status_outcome,
        ).inc()

        validation_results = final_output.get("validation_results", {})
        metrics = validation_results.get("metrics", {})
        precision_mode = final_output.get("precision_mode", "FP16")

        tps_metric = metrics.get("tokens_per_second", 0)
        if tps_metric > 0:
            INFERENCE_THROUGHPUT_GAUGE.labels(
                model_name=final_output["model_name"],
                target_gpu=final_output["target_gpu"],
                target_environment=final_output["target_environment"],
                status=status_outcome,
            ).observe(tps_metric)
            VALIDATION_MATRIX_BENCHMARK_THROUGHPUT_TPS.labels(
                model_name=final_output["model_name"],
                target_gpu=final_output["target_gpu"],
                target_environment=final_output["target_environment"],
                precision_mode=precision_mode,
                status=status_outcome,
            ).observe(tps_metric)

        latency_metric = metrics.get("latency_ms", 0)
        if latency_metric > 0:
            INFERENCE_LATENCY_GAUGE.labels(
                model_name=final_output["model_name"],
                target_gpu=final_output["target_gpu"],
                target_environment=final_output["target_environment"],
                status=status_outcome,
            ).observe(latency_metric)
            VALIDATION_MATRIX_BENCHMARK_LATENCY_MS.labels(
                model_name=final_output["model_name"],
                target_gpu=final_output["target_gpu"],
                target_environment=final_output["target_environment"],
                precision_mode=precision_mode,
                status=status_outcome,
            ).observe(latency_metric)

        if status_outcome != PUBLISH_STATUS:
            failure_stage, failure_reason = _failure_stage_and_reason(final_output)
            FACTORY_FAILURES.labels(
                stage=failure_stage,
                reason=failure_reason,
                model=final_output["model_name"],
            ).inc()

        _record_stage_durations(final_output, status_outcome)
        _record_validation_resource_metrics(final_output)

        return {
            "model": final_output["model_name"],
            "hardware": final_output["target_gpu"],
            "environment": final_output["target_environment"],
            "precision_mode": final_output.get("precision_mode", "FP16"),
            "topology_used": final_output["hardware_topology"],
            "deployment_target": final_output["deployment_target"],
            "precision": final_output.get(
                "precision_result",
                {"profiled": False},
            ),
            "compile_result": final_output.get("compile_result", {}),
            "test_metrics": metrics,
            "stage_durations": final_output.get("stage_durations", {}),
            "final_status": status_outcome,
            "deployable": status_outcome == PUBLISH_STATUS,
            "error_log": final_output.get("error_message", ""),
        }
    except Exception:
        LOGGER.exception("factory activity failed unexpectedly")
        raise
    finally:
        ACTIVE_WORKERS.dec()
