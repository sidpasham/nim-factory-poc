import logging
from typing import Any, Dict

from temporalio import activity

from benchmark_graph import (
    PUBLISH_STATUS,
    llm_gpu_benchmarking_graph_pipeline,
    minimum_tps_threshold,
)
from metrics import (
    ACTIVE_WORKERS,
    BENCHMARK_FAILURES,
    BENCHMARK_PIPELINE_RUNS,
    INFERENCE_LATENCY_GAUGE,
    INFERENCE_THROUGHPUT_GAUGE,
    LOCAL_LLM_BENCHMARK_LATENCY_MS,
    LOCAL_LLM_BENCHMARK_THROUGHPUT_TPS,
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
    validation_metrics = validation_results.get("metrics", {})

    if compile_result.get("success") is False or compile_result.get("error_code"):
        return (
            compile_result.get("stage") or "compile",
            compile_result.get("error_code")
            or compile_result.get("reason")
            or "compile_failed",
        )

    if validation_results.get("success") is False:
        return (
            validation_results.get("stage") or "benchmark",
            validation_results.get("error_code")
            or validation_results.get("reason")
            or "validation_failed",
        )

    tokens_per_second = validation_metrics.get("tokens_per_second")
    if tokens_per_second is not None:
        try:
            if float(tokens_per_second) <= minimum_tps_threshold():
                return "benchmark", "throughput_below_threshold"
        except (TypeError, ValueError):
            return "benchmark", "invalid_throughput_metric"

    return "benchmark", "validation_failed"


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


def _build_pipeline_summary(
    final_output: Dict[str, Any],
    metrics: Dict[str, Any],
    status_outcome: str,
) -> Dict[str, Any]:
    precision_mode = final_output.get("precision_mode", "FP16")
    evaluated_target = {
        "target_gpu": final_output["target_gpu"],
        "target_environment": final_output["target_environment"],
        "precision_mode": precision_mode,
    }
    deployable = status_outcome == PUBLISH_STATUS

    return {
        "model": final_output["model_name"],
        "hardware": final_output["target_gpu"],
        "target_gpu": final_output["target_gpu"],
        "environment": final_output["target_environment"],
        "precision_mode": precision_mode,
        "evaluated_target": evaluated_target,
        "deployable_on": evaluated_target if deployable else None,
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
        "deployable": deployable,
        "error_log": final_output.get("error_message", ""),
    }


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
        final_output = llm_gpu_benchmarking_graph_pipeline.invoke(state)
        status_outcome = final_output.get("status", "Unknown")

        BENCHMARK_PIPELINE_RUNS.labels(
            model_name=final_output["model_name"],
            target_gpu=final_output["target_gpu"],
            target_environment=final_output["target_environment"],
            status=status_outcome,
        ).inc()

        validation_results = final_output.get("validation_results", {})
        metrics = validation_results.get("metrics", {})
        precision_mode = final_output.get("precision_mode", "FP16")
        backend = validation_results.get("backend", "unknown")

        tps_metric = metrics.get("tokens_per_second", 0)
        if tps_metric > 0:
            INFERENCE_THROUGHPUT_GAUGE.labels(
                model_name=final_output["model_name"],
                target_gpu=final_output["target_gpu"],
                target_environment=final_output["target_environment"],
                status=status_outcome,
            ).observe(tps_metric)
            if backend == "local_llama_cpp":
                LOCAL_LLM_BENCHMARK_THROUGHPUT_TPS.labels(
                    model_name=final_output["model_name"],
                    target_gpu=final_output["target_gpu"],
                    target_environment=final_output["target_environment"],
                    precision_mode=precision_mode,
                    status=status_outcome,
                ).observe(tps_metric)
            elif backend == "validation_matrix":
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
            if backend == "local_llama_cpp":
                LOCAL_LLM_BENCHMARK_LATENCY_MS.labels(
                    model_name=final_output["model_name"],
                    target_gpu=final_output["target_gpu"],
                    target_environment=final_output["target_environment"],
                    precision_mode=precision_mode,
                    status=status_outcome,
                ).observe(latency_metric)
            elif backend == "validation_matrix":
                VALIDATION_MATRIX_BENCHMARK_LATENCY_MS.labels(
                    model_name=final_output["model_name"],
                    target_gpu=final_output["target_gpu"],
                    target_environment=final_output["target_environment"],
                    precision_mode=precision_mode,
                    status=status_outcome,
                ).observe(latency_metric)

        if status_outcome != PUBLISH_STATUS:
            failure_stage, failure_reason = _failure_stage_and_reason(final_output)
            BENCHMARK_FAILURES.labels(
                stage=failure_stage,
                reason=failure_reason,
                model=final_output["model_name"],
            ).inc()

        _record_stage_durations(final_output, status_outcome)
        _record_validation_resource_metrics(final_output)

        return _build_pipeline_summary(final_output, metrics, status_outcome)
    except Exception:
        LOGGER.exception("benchmark activity failed unexpectedly")
        raise
    finally:
        ACTIVE_WORKERS.dec()
