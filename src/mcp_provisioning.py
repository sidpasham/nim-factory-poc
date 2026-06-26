from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any, Dict

from local_llm_runtime import (
    LOCAL_LLAMA_CPP_SOURCE,
    LocalLLMBenchmarkRunner,
    LocalLLMRuntimeError,
)
from nim_runtime_client import NIMRuntimeClient
from validation_matrix import (
    ValidationMatrixError,
    evaluate_hardware_fit as matrix_evaluate_hardware_fit,
    normalize_precision_mode,
    simulate_benchmark,
)


CONFIG_DIR = Path(__file__).resolve().parent / "config"
HARDWARE_PROFILES_PATH = CONFIG_DIR / "hardware_profiles.json"
DEPLOYMENT_TARGETS_PATH = CONFIG_DIR / "deployment_targets.json"
NIM_VALIDATION_SOURCE = "nvidia_hosted_nim"
VALIDATION_MATRIX_SOURCE = "validation_matrix"
VALIDATION_MATRIX_MODES = {"validation", "matrix"}
LOCAL_VALIDATION_MODE = "local"
HOSTED_VALIDATION_MODE = "hosted"
VALIDATION_MODE_ENV = "LLM_GPU_BENCHMARKING_VALIDATION_MODE"
DEFAULT_VALIDATION_MODE = LOCAL_VALIDATION_MODE


class UnsupportedProvisioningTargetError(ValueError):
    """Raised when a requested hardware or deployment target is not configured."""


def _load_json_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


@lru_cache(maxsize=1)
def _hardware_profiles() -> Dict[str, Any]:
    return _load_json_config(HARDWARE_PROFILES_PATH)


@lru_cache(maxsize=1)
def _deployment_targets() -> Dict[str, Any]:
    return _load_json_config(DEPLOYMENT_TARGETS_PATH)


class MCPProvisioningServer:
    """Config-backed provisioning facade for hardware topology and hosted NIM validation."""

    @staticmethod
    def _supported_hardware_targets(profiles_config: Dict[str, Any]) -> str:
        targets = [
            profile.get("topology", {}).get("accelerator", profile.get("name", "unknown"))
            for profile in profiles_config.get("profiles", [])
        ]
        return ", ".join(sorted(set(targets)))

    @staticmethod
    def _supported_deployment_targets(targets_config: Dict[str, Any]) -> str:
        targets = [
            target.get("target", {}).get("environment", target.get("name", "unknown"))
            for target in targets_config.get("targets", [])
        ]
        return ", ".join(sorted(set(targets)))

    @staticmethod
    def query_hardware_topology(target_gpu: str) -> dict:
        """Queries infrastructure topology based on config-backed GPU profiles."""
        profiles_config = _hardware_profiles()
        target_gpu_normalized = target_gpu.upper()

        for profile in profiles_config.get("profiles", []):
            match_terms = profile.get("match_any", [])
            if any(term.upper() in target_gpu_normalized for term in match_terms):
                return dict(profile["topology"])

        supported_targets = MCPProvisioningServer._supported_hardware_targets(
            profiles_config
        )
        raise UnsupportedProvisioningTargetError(
            f"unsupported target_gpu '{target_gpu}'. "
            f"Supported hardware targets: {supported_targets}"
        )

    @staticmethod
    def query_deployment_target(target_environment: str) -> dict:
        """Returns deployment target metadata for cloud, on-prem, or Kubernetes."""
        targets_config = _deployment_targets()
        target_environment_normalized = target_environment.upper()

        for target in targets_config.get("targets", []):
            match_terms = target.get("match_any", [])
            if any(term.upper() in target_environment_normalized for term in match_terms):
                return dict(target["target"])

        supported_targets = MCPProvisioningServer._supported_deployment_targets(
            targets_config
        )
        raise UnsupportedProvisioningTargetError(
            f"unsupported target_environment '{target_environment}'. "
            f"Supported deployment targets: {supported_targets}"
        )

    @staticmethod
    def evaluate_hardware_fit(
        topology: dict,
        model_name: str,
        target_gpu: str,
        precision_mode: str = "FP16",
    ) -> dict:
        """Prepares the selected validation backend before benchmark/deploy."""
        validation_mode = MCPProvisioningServer._validation_mode()
        if validation_mode == LOCAL_VALIDATION_MODE:
            return LocalLLMBenchmarkRunner.from_env().prepare_model(
                model_name=model_name,
                target_gpu=target_gpu,
                topology=topology,
                precision_mode=precision_mode,
            )

        return matrix_evaluate_hardware_fit(
            model_name=model_name,
            target_gpu=target_gpu,
            topology=topology,
            precision_mode=precision_mode,
        )

    @staticmethod
    def run_hardware_test_harness(
        topology: dict,
        model_name: str = "",
        target_gpu: str = "",
        precision_mode: str = "FP16",
        compile_result: dict | None = None,
    ) -> dict:
        """Runs benchmark validation with the configured backend."""
        validation_mode = MCPProvisioningServer._validation_mode()
        normalized_precision = normalize_precision_mode(precision_mode)

        if validation_mode == LOCAL_VALIDATION_MODE:
            if not model_name:
                return MCPProvisioningServer._failed_local_validation(
                    "model_name is required for local llama.cpp validation",
                    topology,
                )

            try:
                return LocalLLMBenchmarkRunner.from_env().benchmark_model(
                    model_name=model_name,
                    target_gpu=target_gpu or topology.get("accelerator", "unknown"),
                    topology=topology,
                    precision_mode=normalized_precision,
                    compile_result=compile_result,
                )
            except LocalLLMRuntimeError as exc:
                result = exc.to_result()
                return {
                    "success": False,
                    "backend": LOCAL_LLAMA_CPP_SOURCE,
                    "error_code": result.get("error_code"),
                    "stage": result.get("stage"),
                    "reason": result.get("reason"),
                    "error_message": result.get("error_message"),
                    "metrics": {
                        "tokens_per_second": 0,
                        "error_rate": 1.0,
                        "latency_ms": 0,
                        "output_tokens": 0,
                    },
                    "local_llm": result.get("local_runtime", {}),
                    "gpu_simulation": result.get("gpu_simulation", {}),
                    "target_topology": topology,
                }

        if validation_mode == HOSTED_VALIDATION_MODE:
            return MCPProvisioningServer._run_nvidia_hosted_nim_validation(
                model_name,
                topology,
            )

        if validation_mode not in VALIDATION_MATRIX_MODES:
            return {
                "success": False,
                "backend": "unsupported",
                "error_message": (
                    f"unsupported {VALIDATION_MODE_ENV} '{validation_mode}'. "
                    "Supported values: local, hosted, matrix"
                ),
                "metrics": {
                    "tokens_per_second": 0,
                    "error_rate": 1.0,
                    "latency_ms": 0,
                    "output_tokens": 0,
                },
                "target_topology": topology,
            }

        if not model_name:
            return {
                "success": False,
                "backend": VALIDATION_MATRIX_SOURCE,
                "error_message": "model_name is required for validation matrix",
                "metrics": {
                    "tokens_per_second": 0,
                    "error_rate": 1.0,
                    "latency_ms": 0,
                    "output_tokens": 0,
                },
                "target_topology": topology,
            }

        try:
            compiled = compile_result or MCPProvisioningServer.evaluate_hardware_fit(
                topology=topology,
                model_name=model_name,
                target_gpu=target_gpu or topology.get("accelerator", "unknown"),
                precision_mode=normalized_precision,
            )
            result = simulate_benchmark(
                model_name=model_name,
                target_gpu=target_gpu or topology.get("accelerator", "unknown"),
                topology=topology,
                precision_mode=normalized_precision,
                compile_result=compiled,
            )
            result["target_topology"] = topology
            return result
        except ValidationMatrixError as exc:
            result = exc.to_result()
            return {
                "success": False,
                "backend": VALIDATION_MATRIX_SOURCE,
                "error_code": result.get("error_code"),
                "stage": result.get("stage"),
                "reason": result.get("reason"),
                "error_message": result.get("error_message"),
                "metrics": {
                    "tokens_per_second": 0,
                    "error_rate": 1.0,
                    "latency_ms": 0,
                    "output_tokens": 0,
                },
                "validation_matrix": result,
                "target_topology": topology,
            }

    @staticmethod
    def _run_nvidia_hosted_nim_validation(model_name: str, topology: dict) -> dict:
        if not model_name:
            return {
                "success": False,
                "backend": NIM_VALIDATION_SOURCE,
                "error_message": "model_name is required for remote NIM validation",
                "metrics": {
                    "tokens_per_second": 0,
                    "error_rate": 1.0,
                    "latency_ms": 0,
                    "output_tokens": 0,
                },
                "target_topology": topology,
            }

        try:
            with NIMRuntimeClient.from_env() as client:
                result = client.validate_model(model_name)
                result["target_topology"] = topology
                return result
        except Exception as exc:
            return {
                "success": False,
                "backend": NIM_VALIDATION_SOURCE,
                "error_message": f"remote NIM validation failed: {exc}",
                "metrics": {
                    "tokens_per_second": 0,
                    "error_rate": 1.0,
                    "latency_ms": 0,
                    "output_tokens": 0,
                },
                "target_topology": topology,
            }

    @staticmethod
    def _validation_mode() -> str:
        return os.getenv(
            VALIDATION_MODE_ENV,
            DEFAULT_VALIDATION_MODE,
        ).strip().lower()

    @staticmethod
    def _failed_local_validation(error_message: str, topology: dict) -> dict:
        return {
            "success": False,
            "backend": LOCAL_LLAMA_CPP_SOURCE,
            "error_message": error_message,
            "metrics": {
                "tokens_per_second": 0,
                "error_rate": 1.0,
                "latency_ms": 0,
                "output_tokens": 0,
            },
            "target_topology": topology,
        }
