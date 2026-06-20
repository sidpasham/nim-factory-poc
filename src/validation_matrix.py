import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict


VRAM_INSUFFICIENT_CODE = "VRAM_INSUFFICIENT_EXCEPTION"


@dataclass(frozen=True)
class PrecisionProfile:
    mode: str
    bytes_per_parameter: float
    throughput_multiplier: float
    latency_multiplier: float
    accuracy_score: float
    compile_seconds: float


PRECISION_PROFILES: Dict[str, PrecisionProfile] = {
    "FP16": PrecisionProfile(
        mode="FP16",
        bytes_per_parameter=2.0,
        throughput_multiplier=1.0,
        latency_multiplier=1.0,
        accuracy_score=0.995,
        compile_seconds=0.22,
    ),
    "INT8": PrecisionProfile(
        mode="INT8",
        bytes_per_parameter=1.0,
        throughput_multiplier=1.55,
        latency_multiplier=0.72,
        accuracy_score=0.975,
        compile_seconds=0.32,
    ),
    "INT4": PrecisionProfile(
        mode="INT4",
        bytes_per_parameter=0.5,
        throughput_multiplier=2.35,
        latency_multiplier=0.46,
        accuracy_score=0.935,
        compile_seconds=0.45,
    ),
}

MODEL_PARAMETER_OVERRIDES = {
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": 30.0,
    "minimaxai/minimax-m3": 45.0,
    "deepseek-ai/deepseek-v4-flash": 37.0,
    "z-ai/glm-5.1": 32.0,
    "meta/llama-3.1-8b-instruct": 8.0,
    "minimaxai/minimax-m2.7": 7.0,
}

DEFAULT_PARAMETER_COUNT_BILLION = 7.0
RUNTIME_OVERHEAD_GB = 2.0
RUNTIME_OVERHEAD_MULTIPLIER = 1.12


class ValidationMatrixError(ValueError):
    code = "VALIDATION_MATRIX_ERROR"
    stage = "compile"
    reason = "validation"

    def to_result(self) -> Dict[str, Any]:
        return {
            "success": False,
            "error_code": self.code,
            "stage": self.stage,
            "reason": self.reason,
            "error_message": str(self),
        }


class VRAMInsufficientError(ValidationMatrixError):
    code = VRAM_INSUFFICIENT_CODE
    stage = "compile"
    reason = "oom"

    def __init__(
        self,
        model_name: str,
        target_gpu: str,
        required_vram_gb: float,
        available_vram_gb: float,
        precision_mode: str,
    ):
        self.model_name = model_name
        self.target_gpu = target_gpu
        self.required_vram_gb = required_vram_gb
        self.available_vram_gb = available_vram_gb
        self.precision_mode = precision_mode
        super().__init__(
            f"{self.code}: model {model_name} requires {required_vram_gb:.2f}GB "
            f"VRAM with {precision_mode}, but {target_gpu} exposes "
            f"{available_vram_gb:.2f}GB."
        )

    def to_result(self) -> Dict[str, Any]:
        result = super().to_result()
        result.update(
            {
                "model_name": self.model_name,
                "target_gpu": self.target_gpu,
                "required_vram_gb": round(self.required_vram_gb, 2),
                "available_vram_gb": round(self.available_vram_gb, 2),
                "vram_utilization_ratio": round(
                    self.required_vram_gb / max(self.available_vram_gb, 0.001),
                    4,
                ),
                "precision_mode": self.precision_mode,
            }
        )
        return result


def normalize_precision_mode(mode: Any) -> str:
    normalized = str(mode or "FP16").upper().replace("-", "_").strip()
    aliases = {
        "INT4_AWQ": "INT4",
        "AWQ": "INT4",
        "4BIT": "INT4",
        "8BIT": "INT8",
        "HALF": "FP16",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PRECISION_PROFILES:
        supported = ", ".join(sorted(PRECISION_PROFILES))
        raise ValueError(f"unsupported precision_mode '{mode}'. Supported values: {supported}")
    return normalized


def estimate_parameter_count_billion(model_name: str) -> float:
    model_key = model_name.lower().strip()
    if model_key in MODEL_PARAMETER_OVERRIDES:
        return MODEL_PARAMETER_OVERRIDES[model_key]

    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model_key)
    if match:
        return float(match.group(1))

    return DEFAULT_PARAMETER_COUNT_BILLION


def total_vram_capacity_gb(topology: Dict[str, Any]) -> float:
    vram_per_gpu = float(topology.get("vram_gb", topology.get("memory_gb", 24)))
    gpu_count = int(topology.get("gpu_count", 1))
    return vram_per_gpu * max(gpu_count, 1)


def estimate_required_vram_gb(model_name: str, precision_mode: str) -> float:
    profile = PRECISION_PROFILES[normalize_precision_mode(precision_mode)]
    parameter_count_billion = estimate_parameter_count_billion(model_name)
    weight_footprint_gb = parameter_count_billion * profile.bytes_per_parameter
    return (weight_footprint_gb * RUNTIME_OVERHEAD_MULTIPLIER) + RUNTIME_OVERHEAD_GB


def precision_metadata(mode: str) -> Dict[str, Any]:
    normalized = normalize_precision_mode(mode)
    profile = PRECISION_PROFILES[normalized]
    return {
        "mode": profile.mode,
        "bytes_per_parameter": profile.bytes_per_parameter,
        "throughput_multiplier": profile.throughput_multiplier,
        "latency_multiplier": profile.latency_multiplier,
        "accuracy_score": profile.accuracy_score,
        "compile_seconds": profile.compile_seconds,
    }


def evaluate_hardware_fit(
    model_name: str,
    target_gpu: str,
    topology: Dict[str, Any],
    precision_mode: str,
) -> Dict[str, Any]:
    normalized_precision = normalize_precision_mode(precision_mode)
    parameter_count_billion = estimate_parameter_count_billion(model_name)
    required_vram_gb = estimate_required_vram_gb(model_name, normalized_precision)
    available_vram_gb = total_vram_capacity_gb(topology)

    if required_vram_gb > available_vram_gb:
        raise VRAMInsufficientError(
            model_name=model_name,
            target_gpu=target_gpu,
            required_vram_gb=required_vram_gb,
            available_vram_gb=available_vram_gb,
            precision_mode=normalized_precision,
        )

    return {
        "success": True,
        "stage": "compile",
        "reason": "fit",
        "model_name": model_name,
        "target_gpu": target_gpu,
        "parameter_count_billion": parameter_count_billion,
        "required_vram_gb": round(required_vram_gb, 2),
        "available_vram_gb": round(available_vram_gb, 2),
        "vram_utilization_ratio": round(required_vram_gb / max(available_vram_gb, 0.001), 4),
        "precision_mode": normalized_precision,
        "architecture": topology.get("architecture", topology.get("accelerator", target_gpu)),
    }


def _deterministic_jitter(*parts: Any) -> float:
    material = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    return 0.9 + (bucket / 5000.0)


def simulate_benchmark(
    model_name: str,
    target_gpu: str,
    topology: Dict[str, Any],
    precision_mode: str,
    compile_result: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_precision = normalize_precision_mode(precision_mode)
    profile = PRECISION_PROFILES[normalized_precision]
    parameter_count_billion = float(
        compile_result.get(
            "parameter_count_billion",
            estimate_parameter_count_billion(model_name),
        )
    )
    baseline_tps = float(topology.get("validation_tps_baseline", 450))
    bandwidth_factor = float(topology.get("memory_bandwidth_gbps", 900)) / 900.0
    size_penalty = max(parameter_count_billion / 8.0, 1.0)
    jitter = _deterministic_jitter(model_name, target_gpu, normalized_precision)

    tokens_per_second = (
        baseline_tps
        * bandwidth_factor
        * profile.throughput_multiplier
        * jitter
        / (size_penalty ** 0.72)
    )
    latency_ms = (900.0 / max(tokens_per_second, 1.0)) * 100.0 * profile.latency_multiplier

    return {
        "success": True,
        "backend": "validation_matrix",
        "metrics": {
            "tokens_per_second": round(tokens_per_second, 2),
            "error_rate": 0.0,
            "latency_ms": round(latency_ms, 2),
            "output_tokens": 128,
            "accuracy_score": profile.accuracy_score,
            "required_vram_gb": compile_result["required_vram_gb"],
            "available_vram_gb": compile_result["available_vram_gb"],
            "vram_utilization_ratio": compile_result["vram_utilization_ratio"],
        },
        "validation_matrix": {
            "model_name": model_name,
            "target_gpu": target_gpu,
            "precision_mode": normalized_precision,
            "parameter_count_billion": parameter_count_billion,
            "architecture": compile_result.get("architecture"),
            "required_vram_gb": compile_result["required_vram_gb"],
            "available_vram_gb": compile_result["available_vram_gb"],
        },
    }
