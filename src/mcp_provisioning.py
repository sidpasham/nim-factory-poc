from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict

from nim_runtime_client import NIMRuntimeClient


CONFIG_DIR = Path(__file__).resolve().parent / "config"
HARDWARE_PROFILES_PATH = CONFIG_DIR / "hardware_profiles.json"
DEPLOYMENT_TARGETS_PATH = CONFIG_DIR / "deployment_targets.json"
NIM_VALIDATION_SOURCE = "nvidia_hosted_nim"


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
    """Config-backed provisioning facade for hardware topology and NIM validation."""

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
        """Returns deployment target metadata for cloud, on-prem, or Kubernetes demos."""
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
    def run_hardware_test_harness(topology: dict, model_name: str = "") -> dict:
        """Runs validation against the configured real NIM-compatible endpoint."""
        return MCPProvisioningServer._run_nvidia_hosted_nim_validation(
            model_name,
            topology,
        )

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
