# factory_graph.py
import os
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from mcp_provisioning import MCPProvisioningServer


PUBLISH_STATUS = "NIM_Ready_To_Deploy"
FAILED_STATUS = "Failed"
DEFAULT_MINIMUM_TPS_THRESHOLD = 1200.0
MINIMUM_TPS_THRESHOLD_ENV = "NIM_MINIMUM_TPS_THRESHOLD"


def minimum_tps_threshold() -> float:
    raw_threshold = os.getenv(
        MINIMUM_TPS_THRESHOLD_ENV,
        str(DEFAULT_MINIMUM_TPS_THRESHOLD)
    ).strip()
    if not raw_threshold:
        raise ValueError(f"{MINIMUM_TPS_THRESHOLD_ENV} cannot be empty")

    try:
        return float(raw_threshold)
    except ValueError:
        raise ValueError(
            f"{MINIMUM_TPS_THRESHOLD_ENV} must be a numeric TPS value"
        )


def _format_tps(value: float) -> str:
    return f"{value:g}"


# Define the shared state schema across the factory pipeline
class FactoryState(TypedDict):
    model_name: str
    target_gpu: str
    target_environment: str
    hardware_topology: Dict[str, Any]
    deployment_target: Dict[str, Any]
    validation_results: Dict[str, Any]
    status: str
    error_message: str

# Node 1: Discover Infrastructure via MCP
def discover_infrastructure(state: FactoryState) -> Dict[str, Any]:
    print(
        f"\n[Factory Node] Initiating discovery for {state['target_gpu']} "
        f"in {state['target_environment']}..."
    )
    topology = MCPProvisioningServer.query_hardware_topology(state["target_gpu"])
    deployment_target = MCPProvisioningServer.query_deployment_target(
        state["target_environment"]
    )
    return {
        "hardware_topology": topology,
        "deployment_target": deployment_target,
        "status": "Infrastructure_Discovered",
    }

# Node 2: Run Automated Validation Test Harness
def run_validation_harness(state: FactoryState) -> Dict[str, Any]:
    print("[Factory Node] Running automated integration & performance test harness...")
    results = MCPProvisioningServer.run_hardware_test_harness(
        state["hardware_topology"],
        state["model_name"],
    )
    return {"validation_results": results, "status": "Validation_Completed"}

# Node 3: Process Compilation & Deployment Failure
def handle_failure(state: FactoryState) -> Dict[str, Any]:
    validation_results = state.get("validation_results", {})
    metrics = validation_results.get("metrics", {})
    topology = state.get("hardware_topology", {})

    tokens_per_second = metrics.get("tokens_per_second", 0)
    error_rate = metrics.get("error_rate", "unknown")
    interconnect = topology.get("interconnect", "unknown")
    deployment_target = state.get("deployment_target", {})
    environment = deployment_target.get(
        "environment",
        state.get("target_environment", "unknown")
    )
    threshold = minimum_tps_threshold()

    if validation_results.get("error_message"):
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

    print(f"[Factory Node] Critical Alert: {error_message}")
    return {"status": FAILED_STATUS, "error_message": error_message}

# Node 4: Compile and Publish Verified NIM Service
def compile_and_publish_nim(state: FactoryState) -> Dict[str, Any]:
    print(
        f"[Factory Node] Success! Model {state['model_name']} passed validation "
        f"for {state['target_gpu']} in {state['target_environment']}."
    )
    return {"status": PUBLISH_STATUS}

# Conditional Router: Decides where to direct state based on test harness outputs
def route_validation_results(state: FactoryState) -> str:
    results = state.get("validation_results", {})
    tokens_per_second = results.get("metrics", {}).get("tokens_per_second", 0)
    if results.get("success") and tokens_per_second > minimum_tps_threshold():
        return "publish"
    return "fail"

# Building the workflow graph
workflow = StateGraph(FactoryState)

# Add Nodes
workflow.add_node("discover_infrastructure", discover_infrastructure)
workflow.add_node("run_validation_harness", run_validation_harness)
workflow.add_node("handle_failure", handle_failure)
workflow.add_node("compile_and_publish_nim", compile_and_publish_nim)

# Establish Entrypoint
workflow.set_entry_point("discover_infrastructure")

# Define Edges
workflow.add_edge("discover_infrastructure", "run_validation_harness")

# Conditional Routing Edge
workflow.add_conditional_edges(
    "run_validation_harness",
    route_validation_results,
    {
        "publish": "compile_and_publish_nim",
        "fail": "handle_failure"
    }
)

workflow.add_edge("compile_and_publish_nim", END)
workflow.add_edge("handle_failure", END)

# Compile the execution graph
nim_factory_graph_pipeline = workflow.compile()
