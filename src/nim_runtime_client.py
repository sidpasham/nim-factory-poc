import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx


DEFAULT_BENCHMARK_PROMPT = (
    "Briefly describe why GPU-aware inference validation matters for deploying "
    "large language models."
)
CONFIG_DIR = Path(__file__).resolve().parent / "config"
MODEL_PROFILES_PATH = CONFIG_DIR / "model_profiles.json"


@dataclass(frozen=True)
class NIMModelPayloadProfile:
    model: str
    benchmark_prompt: str = DEFAULT_BENCHMARK_PROMPT
    max_tokens: int = 256
    temperature: float = 0.6
    top_p: Optional[float] = 0.95
    stream: bool = False
    extra_body: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, profile: Dict[str, Any]) -> "NIMModelPayloadProfile":
        model = str(profile.get("model", "")).strip()
        if not model:
            raise ValueError("model profile is missing required field 'model'")

        extra_body = profile.get("extra_body")
        if extra_body is not None and not isinstance(extra_body, dict):
            raise ValueError(f"extra_body for model '{model}' must be a JSON object")

        top_p = profile.get("top_p", 0.95)
        return cls(
            model=model,
            benchmark_prompt=profile.get("benchmark_prompt", DEFAULT_BENCHMARK_PROMPT),
            max_tokens=int(profile.get("max_tokens", 256)),
            temperature=float(profile.get("temperature", 0.6)),
            top_p=float(top_p) if top_p is not None else None,
            stream=bool(profile.get("stream", False)),
            extra_body=extra_body,
        )

    def to_chat_completion_payload(self) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self.benchmark_prompt,
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.extra_body:
            payload.update(self.extra_body)
        return payload

    def metadata(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
            "has_extra_body": bool(self.extra_body),
        }


@dataclass(frozen=True)
class NIMRuntimeConfig:
    base_url: str
    api_key: Optional[str] = None
    timeout_seconds: float = 30.0
    model_profiles: Optional[Dict[str, NIMModelPayloadProfile]] = None

    @classmethod
    def from_env(cls) -> "NIMRuntimeConfig":
        base_url = os.getenv("NIM_BASE_URL", "").strip()
        if not base_url:
            raise ValueError("NIM_BASE_URL is required for NIM validation")

        timeout_seconds = float(os.getenv("NIM_TIMEOUT_SECONDS", "30"))
        model_profiles_path = Path(
            os.getenv("NIM_MODEL_PROFILES_PATH", str(MODEL_PROFILES_PATH))
        )

        return cls(
            base_url=base_url.rstrip("/"),
            api_key=os.getenv("NVIDIA_API_KEY") or None,
            timeout_seconds=timeout_seconds,
            model_profiles=load_model_profiles(model_profiles_path),
        )

    def payload_profile_for(self, model_name: str) -> Optional[NIMModelPayloadProfile]:
        profiles = self.model_profiles or {}
        return profiles.get(model_name)


def load_model_profiles(path: Path) -> Dict[str, NIMModelPayloadProfile]:
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    profiles: Dict[str, NIMModelPayloadProfile] = {}
    for profile_config in config.get("profiles", []):
        profile = NIMModelPayloadProfile.from_dict(profile_config)
        if profile.model in profiles:
            raise ValueError(f"duplicate model profile configured for '{profile.model}'")
        profiles[profile.model] = profile

    if not profiles:
        raise ValueError(f"no model profiles configured in {path}")

    return profiles


class NIMRuntimeClient:
    """Client for validating a model against an OpenAI-compatible NIM endpoint."""

    def __init__(
        self,
        config: NIMRuntimeConfig,
        client: Optional[httpx.Client] = None
    ):
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=config.base_url,
            headers=self._headers(),
            timeout=config.timeout_seconds,
        )

    @classmethod
    def from_env(cls) -> "NIMRuntimeClient":
        return cls(NIMRuntimeConfig.from_env())

    def __enter__(self) -> "NIMRuntimeClient":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _endpoint(self, path: str) -> str:
        base_path = urlparse(self.config.base_url).path.rstrip("/")
        if base_path.endswith("/v1") and path.startswith("/v1/"):
            return path.removeprefix("/v1")
        return path

    def _get(self, path: str) -> httpx.Response:
        return self.client.get(self._endpoint(path))

    def _post(self, path: str, payload: Dict[str, Any]) -> httpx.Response:
        return self.client.post(self._endpoint(path), json=payload)

    def _stream_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage: Dict[str, Any] = {}

        with self.client.stream(
            "POST",
            self._endpoint("/v1/chat/completions"),
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue

                event_data = line.removeprefix("data:").strip()
                if event_data == "[DONE]":
                    break

                chunk = json.loads(event_data)
                usage.update(chunk.get("usage") or {})
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    message = choice.get("message") or {}
                    content = delta.get("content") or message.get("content")
                    reasoning = (
                        delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or message.get("reasoning")
                        or message.get("reasoning_content")
                    )
                    if content:
                        content_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)

        message: Dict[str, Any] = {
            "content": "".join(content_parts)
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)

        return {
            "choices": [
                {
                    "message": message
                }
            ],
            "usage": usage,
        }

    def check_ready(self) -> Dict[str, Any]:
        response = self._get("/v1/health/ready")
        if response.status_code in {404, 405}:
            return {
                "ready": None,
                "health_check": "unsupported",
            }

        response.raise_for_status()
        return {
            "ready": True,
            "health_check": "passed",
        }

    def list_models(self) -> List[str]:
        response = self._get("/v1/models")
        if response.status_code in {404, 405}:
            return []

        response.raise_for_status()
        payload = response.json()
        return [
            model["id"]
            for model in payload.get("data", [])
            if isinstance(model, dict) and model.get("id")
        ]

    def validate_model(self, model_name: str) -> Dict[str, Any]:
        readiness = self.check_ready()

        available_models = self.list_models()
        if available_models and model_name not in available_models:
            return self._failed_validation(
                f"model '{model_name}' was not returned by /v1/models",
                {
                    "available_models": available_models,
                    "model_found": False,
                    **readiness,
                },
            )

        payload_profile = self.config.payload_profile_for(model_name)
        if not payload_profile:
            return self._failed_validation(
                f"no model payload profile configured for '{model_name}'",
                {
                    "configured_models": sorted((self.config.model_profiles or {}).keys()),
                    **readiness,
                },
            )

        payload = payload_profile.to_chat_completion_payload()

        start_time = time.perf_counter()
        if payload_profile.stream:
            completion_payload = self._stream_chat_completion(payload)
        else:
            completion_response = self._post("/v1/chat/completions", payload)
            completion_response.raise_for_status()
            completion_payload = completion_response.json()
        elapsed_seconds = max(time.perf_counter() - start_time, 0.001)

        output_tokens = self._completion_tokens(completion_payload)
        tokens_per_second = output_tokens / elapsed_seconds

        return {
            "success": True,
            "backend": "nvidia_hosted_nim",
            "metrics": {
                "tokens_per_second": round(tokens_per_second, 2),
                "error_rate": 0.0,
                "latency_ms": round(elapsed_seconds * 1000, 2),
                "output_tokens": output_tokens,
            },
            "nim": {
                "model_found": model_name in available_models if available_models else None,
                "available_models": available_models,
                "base_url": self.config.base_url,
                "payload_profile": payload_profile.metadata(),
                **readiness,
            },
        }

    def _completion_tokens(self, payload: Dict[str, Any]) -> int:
        usage = payload.get("usage", {})
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        if completion_tokens:
            return int(completion_tokens)

        message = payload.get("choices", [{}])[0].get("message", {})
        message_content = (
            message.get("content")
            or message.get("reasoning")
            or message.get("reasoning_content")
            or ""
        )
        return max(1, len(message_content.split()))

    def _failed_validation(
        self,
        error_message: str,
        nim_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "backend": "nvidia_hosted_nim",
            "error_message": error_message,
            "metrics": {
                "tokens_per_second": 0,
                "error_rate": 1.0,
                "latency_ms": 0,
                "output_tokens": 0,
            },
            "nim": nim_details or {},
        }
