from functools import lru_cache
import json
import random
from pathlib import Path
from typing import Any, Dict


CONFIG_DIR = Path(__file__).resolve().parent / "config"
HARDWARE_PROFILES_PATH = CONFIG_DIR / "hardware_profiles.json"
VALIDATION_PROFILES_PATH = CONFIG_DIR / "validation_profiles.json"


def _load_json_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


@lru_cache(maxsize=1)
def _hardware_profiles() -> Dict[str, Any]:
    return _load_json_config(HARDWARE_PROFILES_PATH)


@lru_cache(maxsize=1)
def _validation_profiles() -> Dict[str, Any]:
    return _load_json_config(VALIDATION_PROFILES_PATH)


class MCPProvisioningServer:
    """Simulates an MCP Server exposing hardware provisioning and platform APIs."""

    @staticmethod
    def query_hardware_topology(target_gpu: str) -> dict:
        """Queries infrastructure topology based on config-backed GPU profiles."""
        profiles_config = _hardware_profiles()
        target_gpu_normalized = target_gpu.upper()

        for profile in profiles_config.get("profiles", []):
            match_terms = profile.get("match_any", [])
            if any(term.upper() in target_gpu_normalized for term in match_terms):
                return dict(profile["topology"])

        return dict(profiles_config["default_profile"])

    @staticmethod
    def run_hardware_test_harness(topology: dict) -> dict:
        """Runs config-backed validation harness simulation against the environment."""
        validation_config = _validation_profiles()
        interconnect = topology.get("interconnect", "default")
        profile = validation_config.get("profiles", {}).get(
            interconnect,
            validation_config.get("default_profile", {})
        )

        success_rate = profile.get("success_rate", validation_config.get("default_success_rate", 0.85))
        is_successful = random.random() < success_rate
        throughput_range = profile.get("tokens_per_second", {"min": 1000, "max": 1500})
        latency_range = profile.get("latency_ms", {"min": 12.5, "max": 24.0})

        return {
            "success": is_successful,
            "metrics": {
                "tokens_per_second": random.randint(
                    throughput_range["min"],
                    throughput_range["max"]
                ),
                "error_rate": profile.get("success_error_rate", 0.0)
                if is_successful
                else profile.get("failure_error_rate", 0.12),
                "latency_ms": random.uniform(
                    latency_range["min"],
                    latency_range["max"]
                )
            }
        }
