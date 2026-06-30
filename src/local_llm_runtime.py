import hashlib
import fcntl
import json
import logging
import os
import platform
import re
import shlex
import shutil
import ssl
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote

import certifi

from validation_matrix import (
    VRAM_INSUFFICIENT_CODE,
    ValidationMatrixError,
    normalize_precision_mode,
)

LOGGER = logging.getLogger(__name__)

LOCAL_LLAMA_CPP_SOURCE = "local_llama_cpp"
CONFIG_DIR = Path(__file__).resolve().parent / "config"
LOCAL_LLM_PROFILES_PATH = CONFIG_DIR / "local_llm_profiles.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_LLM_ROOT_ENV = "LLM_GPU_BENCHMARKING_LOCAL_LLM_ROOT"
LOCAL_LLM_PROFILES_PATH_ENV = "LLM_GPU_BENCHMARKING_LOCAL_LLM_PROFILES_PATH"
LLAMA_CPP_REPO_URL_ENV = "LLAMA_CPP_REPO_URL"
LLAMA_CPP_GIT_REF_ENV = "LLAMA_CPP_GIT_REF"
LLAMA_CPP_CLI_PATH_ENV = "LLAMA_CPP_CLI_PATH"
LLAMA_CPP_CMAKE_ARGS_ENV = "LLAMA_CPP_CMAKE_ARGS"
LLAMA_CPP_BUILD_JOBS_ENV = "LLAMA_CPP_BUILD_JOBS"
LLAMA_CPP_THREADS_ENV = "LLAMA_CPP_THREADS"
LLAMA_CPP_BENCH_REPETITIONS_ENV = "LLAMA_CPP_BENCH_REPETITIONS"
LOCAL_LLM_DOWNLOAD_TIMEOUT_ENV = "LOCAL_LLM_DOWNLOAD_TIMEOUT_SECONDS"
LOCAL_LLM_BENCHMARK_TIMEOUT_ENV = "LOCAL_LLM_BENCHMARK_TIMEOUT_SECONDS"
LOCAL_LLM_CA_BUNDLE_ENV = "LOCAL_LLM_CA_BUNDLE"
LOCAL_LLM_WARMUP_ON_STARTUP_ENV = "LOCAL_LLM_WARMUP_ON_STARTUP"
LOCAL_LLM_WARMUP_MODELS_ENV = "LOCAL_LLM_WARMUP_MODELS"
LOCAL_LLM_WARMUP_PRECISION_MODES_ENV = "LOCAL_LLM_WARMUP_PRECISION_MODES"
LOCAL_LLM_WARMUP_REQUIRED_ENV = "LOCAL_LLM_WARMUP_REQUIRED"

DEFAULT_LLAMA_CPP_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_LLAMA_CPP_GIT_REF = "master"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".runtime" / "local-llm"
DEFAULT_CMAKE_ARGS = ("-DGGML_METAL=OFF", "-DBUILD_SHARED_LIBS=OFF")
BYTES_PER_GIB = 1024**3
CACHE_MANIFEST_VERSION = 1


class LocalLLMRuntimeError(ValidationMatrixError):
    code = "LOCAL_LLM_RUNTIME_ERROR"
    stage = "compile"
    reason = "local_runtime"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        stage: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self._code = code or self.code
        self._stage = stage or self.stage
        self._reason = reason or self.reason
        self.details = details or {}
        super().__init__(message)

    def to_result(self) -> Dict[str, Any]:
        result = super().to_result()
        result.update(
            {
                "backend": LOCAL_LLAMA_CPP_SOURCE,
                "error_code": self._code,
                "stage": self._stage,
                "reason": self._reason,
            }
        )
        result.update(self.details)
        return result


class LocalLLMVRAMInsufficientError(LocalLLMRuntimeError):
    def __init__(
        self,
        model_name: str,
        target_gpu: str,
        required_vram_gb: float,
        available_vram_gb: float,
        precision_mode: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = (
            f"{VRAM_INSUFFICIENT_CODE}: local model artifact for {model_name} requires "
            f"{required_vram_gb:.2f}GB simulated target VRAM with {precision_mode}, "
            f"but {target_gpu} exposes {available_vram_gb:.2f}GB."
        )
        merged_details = {
            "model_name": model_name,
            "target_gpu": target_gpu,
            "required_vram_gb": round(required_vram_gb, 2),
            "available_vram_gb": round(available_vram_gb, 2),
            "vram_utilization_ratio": round(
                required_vram_gb / max(available_vram_gb, 0.001),
                4,
            ),
            "precision_mode": precision_mode,
        }
        if details:
            merged_details.update(details)
        super().__init__(
            message,
            code=VRAM_INSUFFICIENT_CODE,
            stage="compile",
            reason="oom",
            details=merged_details,
        )


@dataclass(frozen=True)
class LocalModelArtifact:
    filename: str
    quantization: str
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LocalModelArtifact":
        filename = str(config.get("filename", "")).strip()
        if not filename:
            raise ValueError("local model artifact is missing required field 'filename'")

        quantization = str(config.get("quantization", "")).strip()
        if not quantization:
            raise ValueError(f"local model artifact '{filename}' is missing quantization")

        size_bytes = config.get("size_bytes")
        return cls(
            filename=filename,
            quantization=quantization,
            sha256=config.get("sha256"),
            size_bytes=int(size_bytes) if size_bytes is not None else None,
        )


@dataclass(frozen=True)
class LocalLLMProfile:
    model_name: str
    aliases: tuple[str, ...]
    hf_repo: str
    architecture: str
    license: str
    parameter_count_billion: float
    context_size: int
    max_tokens: int
    temperature: float
    benchmark_prompt: str
    artifacts: Dict[str, LocalModelArtifact]

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LocalLLMProfile":
        model_name = str(config.get("model_name", "")).strip()
        hf_repo = str(config.get("hf_repo", "")).strip()
        if not model_name:
            raise ValueError("local model profile is missing required field 'model_name'")
        if not hf_repo:
            raise ValueError(f"local model profile '{model_name}' is missing hf_repo")

        artifacts_config = config.get("artifacts")
        if not isinstance(artifacts_config, dict) or not artifacts_config:
            raise ValueError(f"local model profile '{model_name}' has no artifacts")

        artifacts = {
            normalize_precision_mode(precision): LocalModelArtifact.from_dict(artifact)
            for precision, artifact in artifacts_config.items()
        }

        return cls(
            model_name=model_name,
            aliases=tuple(str(alias).strip() for alias in config.get("aliases", []) if alias),
            hf_repo=hf_repo,
            architecture=str(config.get("architecture", "unknown")),
            license=str(config.get("license", "unknown")),
            parameter_count_billion=float(config["parameter_count_billion"]),
            context_size=int(config.get("context_size", 2048)),
            max_tokens=int(config.get("max_tokens", 128)),
            temperature=float(config.get("temperature", 0.0)),
            benchmark_prompt=str(config.get("benchmark_prompt", "")).strip()
            or "Briefly describe why local LLM inference validation matters.",
            artifacts=artifacts,
        )

    def artifact_for(self, precision_mode: str) -> LocalModelArtifact:
        normalized = normalize_precision_mode(precision_mode)
        if normalized not in self.artifacts:
            supported = ", ".join(sorted(self.artifacts))
            raise LocalLLMRuntimeError(
                (
                    f"model '{self.model_name}' does not configure a {normalized} "
                    f"GGUF artifact. Supported precision modes: {supported}"
                ),
                code="LOCAL_MODEL_PRECISION_UNSUPPORTED",
                reason="model_profile",
            )
        return self.artifacts[normalized]

    def matches(self, model_name: str) -> bool:
        requested = model_name.strip().lower()
        names = {self.model_name.lower(), self.hf_repo.lower()}
        names.update(alias.lower() for alias in self.aliases)
        return requested in names


@dataclass(frozen=True)
class LocalLLMRuntimeConfig:
    root_dir: Path
    profiles_path: Path
    repo_url: str
    repo_ref: str
    cmake_args: tuple[str, ...]
    build_jobs: int
    bench_repetitions: int
    explicit_cli_path: Optional[Path]
    ca_bundle_path: Path
    download_timeout_seconds: float
    benchmark_timeout_seconds: float
    warmup_on_startup: bool
    warmup_models: tuple[str, ...]
    warmup_precision_modes: tuple[str, ...]
    warmup_required: bool

    @classmethod
    def from_env(cls) -> "LocalLLMRuntimeConfig":
        explicit_cli = os.getenv(LLAMA_CPP_CLI_PATH_ENV, "").strip()
        ca_bundle = os.getenv(LOCAL_LLM_CA_BUNDLE_ENV, certifi.where()).strip()
        cmake_args = tuple(
            shlex.split(os.getenv(LLAMA_CPP_CMAKE_ARGS_ENV, " ".join(DEFAULT_CMAKE_ARGS)))
        )
        return cls(
            root_dir=Path(os.getenv(LOCAL_LLM_ROOT_ENV, str(DEFAULT_RUNTIME_ROOT))).expanduser(),
            profiles_path=Path(
                os.getenv(LOCAL_LLM_PROFILES_PATH_ENV, str(LOCAL_LLM_PROFILES_PATH))
            ).expanduser(),
            repo_url=os.getenv(LLAMA_CPP_REPO_URL_ENV, DEFAULT_LLAMA_CPP_REPO_URL).strip(),
            repo_ref=os.getenv(LLAMA_CPP_GIT_REF_ENV, DEFAULT_LLAMA_CPP_GIT_REF).strip(),
            cmake_args=cmake_args,
            build_jobs=max(1, int(os.getenv(LLAMA_CPP_BUILD_JOBS_ENV, str(os.cpu_count() or 4)))),
            bench_repetitions=max(1, int(os.getenv(LLAMA_CPP_BENCH_REPETITIONS_ENV, "1"))),
            explicit_cli_path=Path(explicit_cli).expanduser() if explicit_cli else None,
            ca_bundle_path=Path(ca_bundle).expanduser(),
            download_timeout_seconds=float(os.getenv(LOCAL_LLM_DOWNLOAD_TIMEOUT_ENV, "1800")),
            benchmark_timeout_seconds=float(os.getenv(LOCAL_LLM_BENCHMARK_TIMEOUT_ENV, "900")),
            warmup_on_startup=_parse_bool(os.getenv(LOCAL_LLM_WARMUP_ON_STARTUP_ENV, "true")),
            warmup_models=_parse_csv(os.getenv(LOCAL_LLM_WARMUP_MODELS_ENV, "all")),
            warmup_precision_modes=_parse_csv(
                os.getenv(LOCAL_LLM_WARMUP_PRECISION_MODES_ENV, "INT4")
            ),
            warmup_required=_parse_bool(os.getenv(LOCAL_LLM_WARMUP_REQUIRED_ENV, "true")),
        )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def load_local_model_profiles(path: Path = LOCAL_LLM_PROFILES_PATH) -> Dict[str, LocalLLMProfile]:
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    profiles: Dict[str, LocalLLMProfile] = {}
    for profile_config in config.get("profiles", []):
        profile = LocalLLMProfile.from_dict(profile_config)
        keys = {profile.model_name.lower(), profile.hf_repo.lower()}
        keys.update(alias.lower() for alias in profile.aliases)
        for key in keys:
            if key in profiles:
                raise ValueError(f"duplicate local model profile alias configured for '{key}'")
            profiles[key] = profile

    if not profiles:
        raise ValueError(f"no local LLM profiles configured in {path}")
    return profiles


class LocalLLMBenchmarkRunner:
    """Builds llama.cpp, downloads GGUF artifacts, and measures real local inference."""

    def __init__(self, config: LocalLLMRuntimeConfig):
        self.config = config
        self._profiles: Optional[Dict[str, LocalLLMProfile]] = None

    @classmethod
    def from_env(cls) -> "LocalLLMBenchmarkRunner":
        return cls(LocalLLMRuntimeConfig.from_env())

    def warmup_cache(
        self,
        *,
        models: Optional[Iterable[str]] = None,
        precision_modes: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Prepare runtime and configured model artifacts before serving work."""
        start_time = time.perf_counter()
        llama_cli_path = self.ensure_llama_cpp()
        selected_profiles = self._warmup_profiles(models)
        selected_precision_modes = self._warmup_precision_modes(precision_modes)
        warmed_models = []

        for profile in selected_profiles:
            model_entry = {
                "model_name": profile.model_name,
                "hf_repo": profile.hf_repo,
                "artifacts": [],
            }
            for precision_mode in selected_precision_modes:
                if precision_mode not in profile.artifacts:
                    continue
                artifact = profile.artifact_for(precision_mode)
                model_path = self.ensure_model_artifact(profile, artifact)
                model_entry["artifacts"].append(
                    {
                        "precision_mode": precision_mode,
                        "filename": artifact.filename,
                        "quantization": artifact.quantization,
                        "path": str(model_path),
                        "size_bytes": model_path.stat().st_size,
                    }
                )
            warmed_models.append(model_entry)

        return {
            "success": True,
            "backend": LOCAL_LLAMA_CPP_SOURCE,
            "duration_seconds": round(time.perf_counter() - start_time, 3),
            "llama_cli_path": str(llama_cli_path),
            "llama_bench_path": str(self._llama_bench_path(llama_cli_path)),
            "cache_manifest_path": str(self._cache_manifest_path()),
            "models": warmed_models,
        }

    def cache_status(self) -> Dict[str, Any]:
        manifest = self._read_cache_manifest()
        cli_path = self._default_llama_cli_path()
        bench_path = self._llama_bench_path(cli_path)
        return {
            "cache_root": str(self.config.root_dir),
            "cache_manifest_path": str(self._cache_manifest_path()),
            "llama_cpp_ready": self._llama_cpp_cache_valid(cli_path, bench_path),
            "llama_cli_path": str(cli_path),
            "llama_bench_path": str(bench_path),
            "manifest": manifest,
        }

    def prepare_model(
        self,
        *,
        model_name: str,
        target_gpu: str,
        topology: Dict[str, Any],
        precision_mode: str,
    ) -> Dict[str, Any]:
        normalized_precision = normalize_precision_mode(precision_mode)
        profile = self.profile_for(model_name)
        artifact = profile.artifact_for(normalized_precision)
        llama_cli_path = self.ensure_llama_cpp()
        model_path = self.ensure_model_artifact(profile, artifact)
        model_size_gb = self._path_size_bytes(model_path) / BYTES_PER_GIB
        available_vram_gb = self._total_vram_capacity_gb(topology)
        required_vram_gb = self._simulated_required_vram_gb(
            model_size_gb=model_size_gb,
            profile=profile,
        )
        host_memory_gb = self._host_memory_gb()
        host_required_gb = self._required_host_memory_gb(model_size_gb)

        if host_memory_gb is not None and host_required_gb > host_memory_gb * 0.86:
            raise LocalLLMRuntimeError(
                (
                    f"local host memory is insufficient for {profile.model_name}: "
                    f"requires about {host_required_gb:.2f}GB RAM, host exposes "
                    f"{host_memory_gb:.2f}GB."
                ),
                code="LOCAL_HOST_MEMORY_INSUFFICIENT",
                reason="host_memory",
                details={
                    "model_name": profile.model_name,
                    "required_host_memory_gb": round(host_required_gb, 2),
                    "host_memory_gb": round(host_memory_gb, 2),
                },
            )

        local_runtime = {
            "backend": LOCAL_LLAMA_CPP_SOURCE,
            "compiler": "llama.cpp",
            "compiler_repo_url": self.config.repo_url,
            "compiler_ref": self.config.repo_ref,
            "llama_cli_path": str(llama_cli_path),
            "llama_bench_path": str(self._llama_bench_path(llama_cli_path)),
            "model_repo": profile.hf_repo,
            "model_artifact_path": str(model_path),
            "model_artifact_filename": artifact.filename,
            "model_file_size_gb": round(model_size_gb, 4),
            "quantization": artifact.quantization,
            "host_execution": "cpu",
            "n_gpu_layers": 0,
            "host_platform": platform.platform(),
            "host_machine": platform.machine(),
            "host_memory_gb": round(host_memory_gb, 2) if host_memory_gb else None,
            "required_host_memory_gb": round(host_required_gb, 2),
        }
        gpu_simulation = {
            "simulated": True,
            "uses_physical_gpu": False,
            "scope": "target GPU memory and topology only",
            "target_accelerator": topology.get("accelerator", target_gpu),
            "target_gpu_count": int(topology.get("gpu_count", 1)),
            "target_vram_gb": round(available_vram_gb, 2),
        }

        if required_vram_gb > available_vram_gb:
            raise LocalLLMVRAMInsufficientError(
                model_name=profile.model_name,
                target_gpu=target_gpu,
                required_vram_gb=required_vram_gb,
                available_vram_gb=available_vram_gb,
                precision_mode=normalized_precision,
                details={
                    "local_runtime": local_runtime,
                    "gpu_simulation": gpu_simulation,
                },
            )

        return {
            "success": True,
            "backend": LOCAL_LLAMA_CPP_SOURCE,
            "stage": "compile",
            "reason": "local_model_artifact_ready",
            "model_name": profile.model_name,
            "requested_model_name": model_name,
            "target_gpu": target_gpu,
            "parameter_count_billion": profile.parameter_count_billion,
            "required_vram_gb": round(required_vram_gb, 2),
            "available_vram_gb": round(available_vram_gb, 2),
            "vram_utilization_ratio": round(
                required_vram_gb / max(available_vram_gb, 0.001),
                4,
            ),
            "precision_mode": normalized_precision,
            "architecture": topology.get("architecture", topology.get("accelerator", target_gpu)),
            "model_architecture": profile.architecture,
            "local_runtime": local_runtime,
            "gpu_simulation": gpu_simulation,
        }

    def benchmark_model(
        self,
        *,
        model_name: str,
        target_gpu: str,
        topology: Dict[str, Any],
        precision_mode: str,
        compile_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        compiled = compile_result
        if not compiled or compiled.get("backend") != LOCAL_LLAMA_CPP_SOURCE:
            compiled = self.prepare_model(
                model_name=model_name,
                target_gpu=target_gpu,
                topology=topology,
                precision_mode=precision_mode,
            )

        local_runtime = compiled.get("local_runtime", {})
        llama_cli_path = Path(local_runtime.get("llama_cli_path", ""))
        llama_bench_path = Path(local_runtime.get("llama_bench_path", ""))
        model_path = Path(local_runtime.get("model_artifact_path", ""))
        profile = self.profile_for(model_name)
        threads = self._threads()

        if not llama_cli_path.is_file():
            raise LocalLLMRuntimeError(
                f"llama.cpp CLI binary does not exist at {llama_cli_path}",
                code="LOCAL_LLAMA_CPP_BINARY_MISSING",
                stage="benchmark",
                reason="runtime_binary",
            )
        if not llama_bench_path.is_file():
            raise LocalLLMRuntimeError(
                f"llama.cpp benchmark binary does not exist at {llama_bench_path}",
                code="LOCAL_LLAMA_CPP_BENCH_BINARY_MISSING",
                stage="benchmark",
                reason="runtime_binary",
            )
        if not model_path.is_file():
            raise LocalLLMRuntimeError(
                f"local model artifact does not exist at {model_path}",
                code="LOCAL_MODEL_ARTIFACT_MISSING",
                stage="benchmark",
                reason="model_artifact",
            )

        sample_command = [
            str(llama_cli_path),
            "-m",
            str(model_path),
            "-p",
            profile.benchmark_prompt,
            "-n",
            str(profile.max_tokens),
            "--ctx-size",
            str(profile.context_size),
            "--temp",
            str(profile.temperature),
            "--threads",
            str(threads),
            "--seed",
            "42",
            "--no-display-prompt",
            "--n-gpu-layers",
            "0",
            "--single-turn",
            "--simple-io",
            "--perf",
        ]
        bench_command = [
            str(llama_bench_path),
            "-m",
            str(model_path),
            "-p",
            str(min(profile.context_size, 128)),
            "-n",
            str(profile.max_tokens),
            "-r",
            str(self.config.bench_repetitions),
            "-t",
            str(threads),
            "-ngl",
            "0",
            "-o",
            "json",
        ]

        start_time = time.perf_counter()
        LOGGER.info(
            "Running local llama.cpp sample inference; model=%s precision_mode=%s threads=%s",
            profile.model_name,
            precision_mode,
            threads,
        )
        sample_completed = self._run(
            sample_command,
            timeout=self.config.benchmark_timeout_seconds,
        )
        sample_elapsed_seconds = max(time.perf_counter() - start_time, 0.001)
        LOGGER.info(
            "Running local llama-bench; model=%s precision_mode=%s threads=%s repetitions=%s",
            profile.model_name,
            precision_mode,
            threads,
            self.config.bench_repetitions,
        )
        bench_completed = self._run(
            bench_command,
            timeout=self.config.benchmark_timeout_seconds,
        )
        parsed = self._parse_llama_bench_metrics(bench_completed.stdout)
        generated_text = self._extract_generated_text(
            sample_completed.stdout,
            profile.benchmark_prompt,
        )

        return {
            "success": True,
            "backend": LOCAL_LLAMA_CPP_SOURCE,
            "metrics": {
                "tokens_per_second": round(parsed["tokens_per_second"], 2),
                "error_rate": 0.0,
                "latency_ms": parsed["generation_latency_ms"],
                "output_tokens": parsed["output_tokens"],
                "required_vram_gb": compiled["required_vram_gb"],
                "available_vram_gb": compiled["available_vram_gb"],
                "vram_utilization_ratio": compiled["vram_utilization_ratio"],
                "model_file_size_gb": local_runtime.get("model_file_size_gb"),
                "prompt_eval_tokens_per_second": parsed.get(
                    "prompt_tokens_per_second"
                ),
                "sample_latency_ms": round(sample_elapsed_seconds * 1000, 2),
            },
            "local_llm": {
                **local_runtime,
                "threads": threads,
                "prompt": profile.benchmark_prompt,
                "generated_text_sample": generated_text[:1000],
                "llama_bench": parsed,
                "sample_command": self._redacted_command(sample_command),
                "bench_command": self._redacted_command(bench_command),
            },
            "gpu_simulation": compiled.get("gpu_simulation", {}),
            "target_topology": topology,
        }

    def profile_for(self, model_name: str) -> LocalLLMProfile:
        profiles = self._load_profiles()
        key = model_name.strip().lower()
        if key not in profiles:
            configured = sorted({profile.model_name for profile in profiles.values()})
            raise LocalLLMRuntimeError(
                (
                    f"local model '{model_name}' is not configured. Configure a GGUF "
                    f"profile in {self.config.profiles_path}. Configured models: "
                    f"{', '.join(configured)}"
                ),
                code="LOCAL_MODEL_NOT_CONFIGURED",
                reason="model_profile",
                details={"configured_models": configured},
            )
        return profiles[key]

    def ensure_llama_cpp(self) -> Path:
        if self.config.explicit_cli_path:
            if not self.config.explicit_cli_path.is_file():
                raise LocalLLMRuntimeError(
                    (
                        f"{LLAMA_CPP_CLI_PATH_ENV} points to "
                        f"{self.config.explicit_cli_path}, but that file does not exist"
                    ),
                    code="LOCAL_LLAMA_CPP_BINARY_MISSING",
                    reason="runtime_binary",
                )
            return self.config.explicit_cli_path

        repo_dir = self._llama_cpp_repo_dir()
        cli_path = self._default_llama_cli_path()
        bench_path = self._llama_bench_path(cli_path)
        if self._llama_cpp_cache_valid(cli_path, bench_path):
            return cli_path

        with self._file_lock("llama-cpp-build"):
            if self._llama_cpp_cache_valid(cli_path, bench_path):
                return cli_path

            self.config.root_dir.mkdir(parents=True, exist_ok=True)
            if not repo_dir.exists():
                LOGGER.info(
                    "Cloning llama.cpp source; repo=%s target=%s",
                    self.config.repo_url,
                    repo_dir,
                )
                self._run([
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    self.config.repo_url,
                    str(repo_dir),
                ])

            if self.config.repo_ref:
                LOGGER.info("Checking out llama.cpp ref; ref=%s", self.config.repo_ref)
                self._run(
                    ["git", "fetch", "--depth", "1", "origin", self.config.repo_ref],
                    cwd=repo_dir,
                )
                self._run(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)

            cmake_configure = [
                "cmake",
                "-B",
                "build",
                "-DCMAKE_BUILD_TYPE=Release",
                *self.config.cmake_args,
            ]
            LOGGER.info(
                "Configuring llama.cpp build; path=%s cmake_args=%s",
                repo_dir,
                " ".join(self.config.cmake_args),
            )
            self._run(cmake_configure, cwd=repo_dir)
            LOGGER.info(
                "Building llama.cpp binaries; jobs=%s targets=llama-cli,llama-bench",
                self.config.build_jobs,
            )
            self._run(
                [
                    "cmake",
                    "--build",
                    "build",
                    "--config",
                    "Release",
                    "-j",
                    str(self.config.build_jobs),
                    "--target",
                    "llama-cli",
                    "llama-bench",
                ],
                cwd=repo_dir,
            )
            self._write_llama_cpp_cache_manifest(cli_path, bench_path)

        if not cli_path.is_file() or not bench_path.is_file():
            raise LocalLLMRuntimeError(
                (
                    "llama.cpp build completed but required binaries were not found: "
                    f"llama-cli={cli_path}, llama-bench={bench_path}"
                ),
                code="LOCAL_LLAMA_CPP_BUILD_OUTPUT_MISSING",
                reason="runtime_binary",
            )
        return cli_path

    def ensure_model_artifact(
        self,
        profile: LocalLLMProfile,
        artifact: LocalModelArtifact,
    ) -> Path:
        model_dir = self.config.root_dir / "models" / self._safe_path(profile.hf_repo)
        model_path = model_dir / artifact.filename
        if model_path.is_file():
            self._validate_artifact(model_path, artifact)
            self._write_model_cache_manifest(profile, artifact, model_path)
            return model_path

        with self._file_lock(f"model-{self._safe_path(profile.hf_repo)}-{artifact.filename}"):
            if model_path.is_file():
                self._validate_artifact(model_path, artifact)
                self._write_model_cache_manifest(profile, artifact, model_path)
                return model_path

            model_dir.mkdir(parents=True, exist_ok=True)
            url = self._huggingface_resolve_url(profile.hf_repo, artifact.filename)
            partial_path = model_path.with_suffix(model_path.suffix + ".partial")
            if partial_path.exists():
                partial_path.unlink()

            LOGGER.info(
                "Downloading local GGUF artifact; repo=%s file=%s target=%s",
                profile.hf_repo,
                artifact.filename,
                model_path,
            )
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "llm-gpu-benchmarking-local-runtime/0.1"},
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.download_timeout_seconds,
                    context=self._ssl_context(),
                ) as response, partial_path.open("wb") as output_file:
                    shutil.copyfileobj(response, output_file, length=8 * 1024 * 1024)
            except Exception as exc:
                if partial_path.exists():
                    partial_path.unlink(missing_ok=True)
                raise LocalLLMRuntimeError(
                    f"failed to download model artifact from {url}: {exc}",
                    code="LOCAL_MODEL_DOWNLOAD_FAILED",
                    reason="model_download",
                    details={"model_repo": profile.hf_repo, "model_artifact": artifact.filename},
                ) from exc

            partial_path.replace(model_path)
            self._validate_artifact(model_path, artifact)
            self._write_model_cache_manifest(profile, artifact, model_path)
            return model_path

    def _load_profiles(self) -> Dict[str, LocalLLMProfile]:
        if self._profiles is None:
            self._profiles = load_local_model_profiles(self.config.profiles_path)
        return self._profiles

    def _validate_artifact(self, model_path: Path, artifact: LocalModelArtifact) -> None:
        if artifact.size_bytes is not None and model_path.stat().st_size != artifact.size_bytes:
            raise LocalLLMRuntimeError(
                (
                    f"model artifact size mismatch for {model_path}: expected "
                    f"{artifact.size_bytes} bytes, got {model_path.stat().st_size}"
                ),
                code="LOCAL_MODEL_ARTIFACT_SIZE_MISMATCH",
                reason="model_artifact",
            )

        if artifact.sha256:
            digest = hashlib.sha256()
            with model_path.open("rb") as model_file:
                for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual.lower() != artifact.sha256.lower():
                raise LocalLLMRuntimeError(
                    (
                        f"model artifact checksum mismatch for {model_path}: expected "
                        f"{artifact.sha256}, got {actual}"
                    ),
                    code="LOCAL_MODEL_ARTIFACT_CHECKSUM_MISMATCH",
                    reason="model_artifact",
                )

    def _warmup_profiles(self, models: Optional[Iterable[str]]) -> list[LocalLLMProfile]:
        profiles_by_alias = self._load_profiles()
        unique_profiles = {
            profile.model_name: profile
            for profile in profiles_by_alias.values()
        }

        selected = tuple(models) if models is not None else self.config.warmup_models
        if not selected or any(item.lower() == "none" for item in selected):
            return []
        if any(item.lower() == "all" for item in selected):
            return list(unique_profiles.values())

        profiles: list[LocalLLMProfile] = []
        for model_name in selected:
            profiles.append(self.profile_for(model_name))
        return profiles

    def _warmup_precision_modes(
        self,
        precision_modes: Optional[Iterable[str]],
    ) -> tuple[str, ...]:
        selected = tuple(precision_modes) if precision_modes is not None else self.config.warmup_precision_modes
        if not selected:
            return ("INT4",)
        if any(item.lower() == "all" for item in selected):
            return tuple(sorted(PRECISION for PRECISION in ("FP16", "INT8", "INT4")))
        return tuple(normalize_precision_mode(mode) for mode in selected)

    def _run(
        self,
        command: Iterable[str],
        *,
        cwd: Optional[Path] = None,
        timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess[str]:
        args = [str(part) for part in command]
        try:
            completed = subprocess.run(
                args,
                cwd=str(cwd) if cwd else None,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise LocalLLMRuntimeError(
                f"required executable '{args[0]}' was not found in PATH",
                code="LOCAL_RUNTIME_EXECUTABLE_MISSING",
                reason="runtime_binary",
                details={"command": args},
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LocalLLMRuntimeError(
                f"command timed out after {timeout} seconds: {' '.join(args)}",
                code="LOCAL_RUNTIME_COMMAND_TIMEOUT",
                stage="benchmark",
                reason="timeout",
                details={"command": args},
            ) from exc

        if completed.returncode != 0:
            raise LocalLLMRuntimeError(
                (
                    f"command failed with exit code {completed.returncode}: "
                    f"{' '.join(args)}\n{self._command_tail(completed)}"
                ),
                code="LOCAL_RUNTIME_COMMAND_FAILED",
                stage="benchmark" if args and "llama-cli" in Path(args[0]).name else "compile",
                reason="command_failed",
                details={"command": self._redacted_command(args)},
            )
        return completed

    def _parse_llama_bench_metrics(self, stdout: str) -> Dict[str, Any]:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LocalLLMRuntimeError(
                "llama-bench completed but did not emit valid JSON metrics",
                code="LOCAL_LLAMA_BENCH_JSON_PARSE_FAILED",
                stage="benchmark",
                reason="metrics_parse",
                details={"stdout_tail": stdout[-4000:]},
            ) from exc

        if not isinstance(payload, list):
            raise LocalLLMRuntimeError(
                "llama-bench JSON metrics were not a list",
                code="LOCAL_LLAMA_BENCH_JSON_SHAPE_INVALID",
                stage="benchmark",
                reason="metrics_parse",
            )

        generation = next((item for item in payload if int(item.get("n_gen", 0)) > 0), None)
        prompt = next((item for item in payload if int(item.get("n_prompt", 0)) > 0), None)
        if not generation:
            raise LocalLLMRuntimeError(
                "llama-bench JSON did not include a generation benchmark row",
                code="LOCAL_LLAMA_BENCH_GENERATION_MISSING",
                stage="benchmark",
                reason="metrics_parse",
                details={"benchmark_rows": payload},
            )

        parsed = {
            "tokens_per_second": float(generation["avg_ts"]),
            "generation_latency_ms": round(float(generation["avg_ns"]) / 1_000_000.0, 2),
            "output_tokens": int(generation["n_gen"]),
            "generation_repetitions": len(generation.get("samples_ts", [])),
            "generation_samples_tps": generation.get("samples_ts", []),
            "model_type": generation.get("model_type"),
            "model_size_bytes": generation.get("model_size"),
            "model_n_params": generation.get("model_n_params"),
            "cpu_info": generation.get("cpu_info"),
            "gpu_info": generation.get("gpu_info"),
            "backends": generation.get("backends"),
            "n_gpu_layers": generation.get("n_gpu_layers"),
        }
        if prompt:
            parsed.update(
                {
                    "prompt_tokens_per_second": round(float(prompt["avg_ts"]), 2),
                    "prompt_latency_ms": round(float(prompt["avg_ns"]) / 1_000_000.0, 2),
                    "prompt_tokens": int(prompt["n_prompt"]),
                    "prompt_repetitions": len(prompt.get("samples_ts", [])),
                    "prompt_samples_tps": prompt.get("samples_ts", []),
                }
            )
        return parsed

    def _extract_generated_text(self, stdout: str, prompt: str) -> str:
        text = stdout.strip()
        prompt_marker = f"> {prompt}"
        if prompt_marker in text:
            text = text.split(prompt_marker, 1)[1].lstrip()

        for marker in ("\n[ Prompt:", "\nExiting..."):
            if marker in text:
                text = text.split(marker, 1)[0].rstrip()

        return text or stdout.strip()[-1000:]

    def _threads(self) -> int:
        configured = os.getenv(LLAMA_CPP_THREADS_ENV, "").strip()
        if configured:
            return max(1, int(configured))
        return max(1, os.cpu_count() or 1)

    def _host_memory_gb(self) -> Optional[float]:
        try:
            if platform.system() == "Darwin":
                try:
                    raw = subprocess.check_output(
                        ["sysctl", "-n", "hw.memsize"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    ).strip()
                    return int(raw) / BYTES_PER_GIB
                except Exception:
                    return self._sysconf_memory_gb()

            return self._sysconf_memory_gb()
        except Exception:
            return None

    def _sysconf_memory_gb(self) -> float:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / BYTES_PER_GIB

    def _required_host_memory_gb(self, model_size_gb: float) -> float:
        return (model_size_gb * 1.25) + 1.0

    def _simulated_required_vram_gb(
        self,
        *,
        model_size_gb: float,
        profile: LocalLLMProfile,
    ) -> float:
        kv_cache_gb = max(0.5, profile.context_size * profile.parameter_count_billion * 0.00004)
        return (model_size_gb * 1.18) + kv_cache_gb

    def _total_vram_capacity_gb(self, topology: Dict[str, Any]) -> float:
        vram_per_gpu = float(topology.get("vram_gb", topology.get("memory_gb", 24)))
        gpu_count = int(topology.get("gpu_count", 1))
        return vram_per_gpu * max(gpu_count, 1)

    def _path_size_bytes(self, path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())

    def _llama_cpp_repo_dir(self) -> Path:
        return self.config.root_dir / "llama.cpp"

    def _default_llama_cli_path(self) -> Path:
        return self._llama_cpp_repo_dir() / "build" / "bin" / "llama-cli"

    def _llama_bench_path(self, llama_cli_path: Path) -> Path:
        return llama_cli_path.parent / "llama-bench"

    def _cache_manifest_path(self) -> Path:
        return self.config.root_dir / "cache_manifest.json"

    def _llama_cpp_cache_key(self) -> str:
        material = json.dumps(
            {
                "repo_url": self.config.repo_url,
                "repo_ref": self.config.repo_ref,
                "cmake_args": list(self.config.cmake_args),
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _llama_cpp_cache_valid(self, cli_path: Path, bench_path: Path) -> bool:
        if not cli_path.is_file() or not bench_path.is_file():
            return False
        manifest = self._read_cache_manifest()
        llama_manifest = manifest.get("llama_cpp", {})
        return llama_manifest.get("cache_key") == self._llama_cpp_cache_key()

    def _read_cache_manifest(self) -> Dict[str, Any]:
        manifest_path = self._cache_manifest_path()
        if not manifest_path.is_file():
            return {
                "schema_version": CACHE_MANIFEST_VERSION,
                "llama_cpp": {},
                "models": {},
            }

        try:
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except json.JSONDecodeError as exc:
            raise LocalLLMRuntimeError(
                f"cache manifest is invalid JSON: {manifest_path}",
                code="LOCAL_LLM_CACHE_MANIFEST_INVALID",
                reason="cache_manifest",
            ) from exc

        manifest.setdefault("schema_version", CACHE_MANIFEST_VERSION)
        manifest.setdefault("llama_cpp", {})
        manifest.setdefault("models", {})
        return manifest

    def _write_cache_manifest(self, manifest: Dict[str, Any]) -> None:
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        manifest["schema_version"] = CACHE_MANIFEST_VERSION
        temp_path = self._cache_manifest_path().with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")
        temp_path.replace(self._cache_manifest_path())

    def _write_llama_cpp_cache_manifest(self, cli_path: Path, bench_path: Path) -> None:
        manifest = self._read_cache_manifest()
        manifest["llama_cpp"] = {
            "cache_key": self._llama_cpp_cache_key(),
            "repo_url": self.config.repo_url,
            "repo_ref": self.config.repo_ref,
            "cmake_args": list(self.config.cmake_args),
            "llama_cli_path": str(cli_path),
            "llama_bench_path": str(bench_path),
            "llama_cli_size_bytes": cli_path.stat().st_size if cli_path.is_file() else None,
            "llama_bench_size_bytes": bench_path.stat().st_size if bench_path.is_file() else None,
            "updated_at_epoch_seconds": round(time.time(), 3),
        }
        self._write_cache_manifest(manifest)

    def _write_model_cache_manifest(
        self,
        profile: LocalLLMProfile,
        artifact: LocalModelArtifact,
        model_path: Path,
    ) -> None:
        manifest = self._read_cache_manifest()
        models = manifest.setdefault("models", {})
        key = f"{profile.hf_repo}/{artifact.filename}"
        models[key] = {
            "model_name": profile.model_name,
            "hf_repo": profile.hf_repo,
            "filename": artifact.filename,
            "quantization": artifact.quantization,
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "configured_size_bytes": artifact.size_bytes,
            "configured_sha256": artifact.sha256,
            "updated_at_epoch_seconds": round(time.time(), 3),
        }
        self._write_cache_manifest(manifest)

    def _safe_path(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "--", value).strip("-") or "model"

    def _huggingface_resolve_url(self, repo: str, filename: str) -> str:
        return f"https://huggingface.co/{repo}/resolve/main/{quote(filename, safe='/')}?download=true"

    def _ssl_context(self) -> ssl.SSLContext:
        if not self.config.ca_bundle_path.is_file():
            raise LocalLLMRuntimeError(
                (
                    f"{LOCAL_LLM_CA_BUNDLE_ENV} points to {self.config.ca_bundle_path}, "
                    "but that CA bundle file does not exist"
                ),
                code="LOCAL_MODEL_DOWNLOAD_CA_BUNDLE_MISSING",
                reason="model_download",
            )
        return ssl.create_default_context(cafile=str(self.config.ca_bundle_path))

    def _command_tail(self, completed: subprocess.CompletedProcess[str]) -> str:
        output = "\n".join(
            part
            for part in [
                completed.stdout[-2000:] if completed.stdout else "",
                completed.stderr[-4000:] if completed.stderr else "",
            ]
            if part
        )
        return output.strip()

    def _redacted_command(self, command: Iterable[str]) -> list[str]:
        return [str(part) for part in command]

    @contextmanager
    def _file_lock(self, name: str):
        lock_dir = self.config.root_dir / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{self._safe_path(name)}.lock"
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
