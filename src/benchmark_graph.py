import logging
import os
import time
from typing import Any, Dict, TypedDict

from langgraph.graph import END, StateGraph

from mcp_provisioning import MCPProvisioningServer
from validation_matrix import (
    ValidationMatrixError,
    normalize_precision_mode,
    precision_metadata,
)


LOGGER = logging.getLogger(__name__)

PUBLISH_STATUS = "Model_Service_Ready_To_Deploy"
FAILED_STATUS = "Failed"
DEFAULT_MINIMUM_TPS_THRESHOLD = 100.0
MINIMUM_TPS_THRESHOLD_ENV = "LLM_GPU_BENCHMARKING_MINIMUM_TPS_THRESHOLD"


def minimum_tps_threshold() -> float:
    raw_threshold = os.getenv(
        MINIMUM_TPS_THRESHOLD_ENV,
        str(DEFAULT_MINIMUM_TPS_THRESHOLD),
    ).strip()
    if not raw_threshold:
        raise ValueError(f"{MINIMUM_TPS_THRESHOLD_ENV} cannot be empty")

    try:
        return float(raw_threshold)
    except ValueError:
        raise ValueError(f"{MINIMUM_TPS_THRESHOLD_ENV} must be a numeric TPS value")


def _format_tps(value: float) -> str:
    return f"{value:g}"


class BenchmarkState(TypedDict, total=False):
    model_name: str
    target_gpu: str
    target_environment: str
    precision_mode: str
    hardware_topology: Dict[str, Any]
    deployment_target: Dict[str, Any]
    precision_result: Dict[str, Any]
    compile_result: Dict[str, Any]
    validation_results: Dict[str, Any]
    stage_durations: Dict[str, float]
    status: str
    error_message: str


def _stage_update(
    state: BenchmarkState,
    stage: str,
    start_time: float,
    update: Dict[str, Any],
) -> Dict[str, Any]:
    stage_durations = dict(state.get("stage_durations", {}))
    stage_durations[stage] = round(time.perf_counter() - start_time, 6)
    update["stage_durations"] = stage_durations
    return update


def discover_infrastructure(state: BenchmarkState) -> Dict[str, Any]:
    start_time = time.perf_counter()
    LOGGER.info(
        "discovering infrastructure for model=%s hardware=%s environment=%s",
        state["model_name"],
        state["target_gpu"],
        state["target_environment"],
    )
    topology = MCPProvisioningServer.query_hardware_topology(state["target_gpu"])
    deployment_target = MCPProvisioningServer.query_deployment_target(
        state["target_environment"]
    )
    return _stage_update(
        state,
        "discover",
        start_time,
        {
            "hardware_topology": topology,
            "deployment_target": deployment_target,
            "status": "Infrastructure_Discovered",
        },
    )


def route_precision_profile(state: BenchmarkState) -> str:
    mode = normalize_precision_mode(state.get("precision_mode", "FP16"))
    if mode == "FP16":
        return "skip"
    return "profile"


def record_precision_profile(state: BenchmarkState) -> Dict[str, Any]:
    start_time = time.perf_counter()
    mode = normalize_precision_mode(state.get("precision_mode", "FP16"))
    metadata = precision_metadata(mode)
    LOGGER.info(
        "recording precision profile mode=%s model=%s",
        mode,
        state["model_name"],
    )
    return _stage_update(
        state,
        "precision_profile",
        start_time,
        {
            "precision_mode": mode,
            "precision_result": {
                "profiled": True,
                **metadata,
            },
            "status": "Precision_Profile_Recorded",
        },
    )


def compile_validation_matrix_plan(state: BenchmarkState) -> Dict[str, Any]:
    start_time = time.perf_counter()
    mode = normalize_precision_mode(state.get("precision_mode", "FP16"))

    try:
        compile_result = MCPProvisioningServer.evaluate_hardware_fit(
            topology=state["hardware_topology"],
            model_name=state["model_name"],
            target_gpu=state["target_gpu"],
            precision_mode=mode,
        )
        LOGGER.info(
            "validation matrix compile fit succeeded model=%s hardware=%s precision_mode=%s required_vram=%sGB capacity=%sGB",
            state["model_name"],
            state["target_gpu"],
            mode,
            compile_result["required_vram_gb"],
            compile_result["available_vram_gb"],
        )
        return _stage_update(
            state,
            "compile",
            start_time,
            {
                "precision_mode": mode,
                "compile_result": compile_result,
                "status": "Compilation_Completed",
            },
        )
    except ValidationMatrixError as exc:
        compile_result = exc.to_result()
        LOGGER.warning(
            "validation matrix compile failed model=%s hardware=%s precision_mode=%s error=%s",
            state["model_name"],
            state["target_gpu"],
            mode,
            compile_result["error_message"],
        )
        return _stage_update(
            state,
            "compile",
            start_time,
            {
                "precision_mode": mode,
                "compile_result": compile_result,
                "validation_results": {
                    "success": False,
                    "backend": "validation_matrix",
                    "error_code": compile_result.get("error_code"),
                    "stage": compile_result.get("stage", "compile"),
                    "reason": compile_result.get("reason", "validation"),
                    "error_message": compile_result["error_message"],
                    "metrics": {
                        "tokens_per_second": 0,
                        "error_rate": 1.0,
                        "latency_ms": 0,
                        "output_tokens": 0,
                    },
                    "validation_matrix": compile_result,
                    "target_topology": state["hardware_topology"],
                },
                "status": "Compilation_Failed",
                "error_message": compile_result["error_message"],
            },
        )


def route_compile_results(state: BenchmarkState) -> str:
    compile_result = state.get("compile_result", {})
    if compile_result.get("success"):
        return "benchmark"
    return "fail"


def run_validation_harness(state: BenchmarkState) -> Dict[str, Any]:
    start_time = time.perf_counter()
    mode = normalize_precision_mode(state.get("precision_mode", "FP16"))
    LOGGER.info(
        "running benchmark harness model=%s hardware=%s precision_mode=%s",
        state["model_name"],
        state["target_gpu"],
        mode,
    )
    results = MCPProvisioningServer.run_hardware_test_harness(
        state["hardware_topology"],
        state["model_name"],
        target_gpu=state["target_gpu"],
        precision_mode=mode,
        compile_result=state.get("compile_result"),
    )
    return _stage_update(
        state,
        "benchmark",
        start_time,
        {
            "validation_results": results,
            "status": "Benchmark_Completed",
        },
    )


def handle_failure(state: BenchmarkState) -> Dict[str, Any]:
    start_time = time.perf_counter()
    validation_results = state.get("validation_results", {})
    compile_result = state.get("compile_result", {})
    metrics = validation_results.get("metrics", {})
    topology = state.get("hardware_topology", {})

    tokens_per_second = metrics.get("tokens_per_second", 0)
    error_rate = metrics.get("error_rate", "unknown")
    interconnect = topology.get("interconnect", "unknown")
    deployment_target = state.get("deployment_target", {})
    environment = deployment_target.get(
        "environment",
        state.get("target_environment", "unknown"),
    )
    threshold = minimum_tps_threshold()

    if compile_result.get("error_message"):
        failure_reason = compile_result["error_message"]
    elif validation_results.get("error_message"):
        failure_reason = validation_results["error_message"]
    elif not validation_results.get("success"):
        failure_reason = "hardware validation harness reported an unsuccessful run"
    else:
        failure_reason = (
            f"throughput {tokens_per_second} TPS did not exceed "
            f"the {_format_tps(threshold)} TPS threshold"
        )

    error_message = (
        f"{failure_reason} for model {state['model_name']} on {state['target_gpu']} "
        f"in {environment} ({interconnect}; error_rate={error_rate})."
    )

    LOGGER.warning("benchmark pipeline failed: %s", error_message)
    return _stage_update(
        state,
        "failure",
        start_time,
        {
            "status": FAILED_STATUS,
            "error_message": error_message,
        },
    )


def compile_and_publish_model_service(state: BenchmarkState) -> Dict[str, Any]:
    start_time = time.perf_counter()
    LOGGER.info(
        "model passed validation model=%s hardware=%s environment=%s precision_mode=%s",
        state["model_name"],
        state["target_gpu"],
        state["target_environment"],
        state.get("precision_mode", "FP16"),
    )
    return _stage_update(
        state,
        "publish",
        start_time,
        {"status": PUBLISH_STATUS},
    )


def route_validation_results(state: BenchmarkState) -> str:
    results = state.get("validation_results", {})
    tokens_per_second = results.get("metrics", {}).get("tokens_per_second", 0)
    if results.get("success") and tokens_per_second > minimum_tps_threshold():
        return "publish"
    return "fail"


workflow = StateGraph(BenchmarkState)
workflow.add_node("discover_infrastructure", discover_infrastructure)
workflow.add_node("record_precision_profile", record_precision_profile)
workflow.add_node("compile_validation_matrix_plan", compile_validation_matrix_plan)
workflow.add_node("run_validation_harness", run_validation_harness)
workflow.add_node("handle_failure", handle_failure)
workflow.add_node("compile_and_publish_model_service", compile_and_publish_model_service)

workflow.set_entry_point("discover_infrastructure")
workflow.add_conditional_edges(
    "discover_infrastructure",
    route_precision_profile,
    {
        "profile": "record_precision_profile",
        "skip": "compile_validation_matrix_plan",
    },
)
workflow.add_edge("record_precision_profile", "compile_validation_matrix_plan")
workflow.add_conditional_edges(
    "compile_validation_matrix_plan",
    route_compile_results,
    {
        "benchmark": "run_validation_harness",
        "fail": "handle_failure",
    },
)
workflow.add_conditional_edges(
    "run_validation_harness",
    route_validation_results,
    {
        "publish": "compile_and_publish_model_service",
        "fail": "handle_failure",
    },
)
workflow.add_edge("compile_and_publish_model_service", END)
workflow.add_edge("handle_failure", END)

llm_gpu_benchmarking_graph_pipeline = workflow.compile()
