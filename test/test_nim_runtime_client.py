import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from nim_runtime_client import (  # noqa: E402
    NIMModelPayloadProfile,
    NIMRuntimeClient,
    NIMRuntimeConfig,
)


def json_from_request(request: httpx.Request):
    import json

    return json.loads(request.content.decode("utf-8"))


class NIMRuntimeClientTests(unittest.TestCase):
    def payload_profiles(self):
        return {
            "meta/llama-test": NIMModelPayloadProfile(
                model="meta/llama-test",
                max_tokens=16,
                temperature=0.0,
                top_p=None,
            ),
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": NIMModelPayloadProfile(
                model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                max_tokens=256,
                temperature=0.6,
                top_p=0.95,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": True
                    },
                    "reasoning_budget": 16384
                },
            ),
            "minimaxai/minimax-m3": NIMModelPayloadProfile(
                model="minimaxai/minimax-m3",
                max_tokens=512,
                temperature=1.0,
                top_p=0.95,
            ),
            "deepseek-ai/deepseek-v4-flash": NIMModelPayloadProfile(
                model="deepseek-ai/deepseek-v4-flash",
                max_tokens=512,
                temperature=1.0,
                top_p=0.95,
                extra_body={
                    "chat_template_kwargs": {
                        "thinking": True,
                        "reasoning_effort": "high"
                    }
                },
            ),
            "z-ai/glm-5.1": NIMModelPayloadProfile(
                model="z-ai/glm-5.1",
                max_tokens=16384,
                temperature=1.0,
                top_p=1.0,
                stream=True,
            ),
            "meta/llama-3.1-8b-instruct": NIMModelPayloadProfile(
                model="meta/llama-3.1-8b-instruct",
                max_tokens=1024,
                temperature=0.2,
                top_p=0.7,
            ),
            "minimaxai/minimax-m2.7": NIMModelPayloadProfile(
                model="minimaxai/minimax-m2.7",
                max_tokens=8192,
                temperature=1.0,
                top_p=0.95,
            ),
        }

    def test_validate_model_measures_chat_completion_tps(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.url.path == "/v1/health/ready":
                return httpx.Response(200, json={"status": "ready"})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "meta/llama-test"}]})
            if request.url.path == "/v1/chat/completions":
                return httpx.Response(200, json={
                    "choices": [
                        {
                            "message": {
                                "content": "NIM validation completed."
                            }
                        }
                    ],
                    "usage": {
                        "completion_tokens": 16
                    }
                })
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        http_client = httpx.Client(
            base_url="http://nim.local",
            transport=transport,
            headers={"Authorization": "Bearer test-token"},
        )
        config = NIMRuntimeConfig(
            base_url="http://nim.local",
            api_key="test-token",
            model_profiles=self.payload_profiles(),
        )

        result = NIMRuntimeClient(config, client=http_client).validate_model("meta/llama-test")

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "nvidia_hosted_nim")
        self.assertEqual(result["metrics"]["output_tokens"], 16)
        self.assertGreater(result["metrics"]["tokens_per_second"], 0)
        self.assertGreater(result["metrics"]["latency_ms"], 0)
        self.assertEqual(requests, [
            ("GET", "/v1/health/ready"),
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
        ])

    def test_validate_model_fails_when_model_is_not_loaded(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/health/ready":
                return httpx.Response(200, json={"status": "ready"})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "other-model"}]})
            return httpx.Response(500)

        http_client = httpx.Client(
            base_url="http://nim.local",
            transport=httpx.MockTransport(handler),
        )
        config = NIMRuntimeConfig(base_url="http://nim.local")

        result = NIMRuntimeClient(config, client=http_client).validate_model("meta/llama-test")

        self.assertFalse(result["success"])
        self.assertEqual(result["backend"], "nvidia_hosted_nim")
        self.assertEqual(result["metrics"]["tokens_per_second"], 0)
        self.assertIn("was not returned by /v1/models", result["error_message"])

    def test_validate_model_supports_nvidia_hosted_base_url_and_extra_body(self):
        requests = []
        payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.url.path == "/v1/health/ready":
                return httpx.Response(404)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={
                    "data": [
                        {
                            "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
                        }
                    ]
                })
            if request.url.path == "/v1/chat/completions":
                payloads.append(json_from_request(request))
                return httpx.Response(200, json={
                    "choices": [
                        {
                            "message": {
                                "content": "Validation completed.",
                                "reasoning_content": "Short reasoning trace."
                            }
                        }
                    ],
                    "usage": {
                        "completion_tokens": 12
                    }
                })
            return httpx.Response(500)

        http_client = httpx.Client(
            base_url="https://integrate.api.nvidia.com/v1",
            transport=httpx.MockTransport(handler),
        )
        config = NIMRuntimeConfig(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="test-token",
            model_profiles=self.payload_profiles(),
        )

        result = NIMRuntimeClient(config, client=http_client).validate_model(
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        )

        self.assertTrue(result["success"])
        self.assertIsNone(result["nim"]["ready"])
        self.assertEqual(result["nim"]["health_check"], "unsupported")
        self.assertEqual(requests, [
            ("GET", "/v1/health/ready"),
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
        ])
        self.assertEqual(payloads[0]["model"], "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
        self.assertEqual(payloads[0]["temperature"], 0.6)
        self.assertEqual(payloads[0]["top_p"], 0.95)
        self.assertEqual(payloads[0]["max_tokens"], 256)
        self.assertEqual(payloads[0]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(payloads[0]["reasoning_budget"], 16384)

    def test_validate_model_supports_minimax_payload_profile(self):
        payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/health/ready":
                return httpx.Response(404)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={
                    "data": [
                        {
                            "id": "minimaxai/minimax-m3"
                        }
                    ]
                })
            if request.url.path == "/v1/chat/completions":
                payloads.append(json_from_request(request))
                return httpx.Response(200, json={
                    "choices": [
                        {
                            "message": {
                                "content": "Validation completed."
                            }
                        }
                    ],
                    "usage": {
                        "completion_tokens": 10
                    }
                })
            return httpx.Response(500)

        http_client = httpx.Client(
            base_url="https://integrate.api.nvidia.com/v1",
            transport=httpx.MockTransport(handler),
        )
        config = NIMRuntimeConfig(
            base_url="https://integrate.api.nvidia.com/v1",
            model_profiles=self.payload_profiles(),
        )

        result = NIMRuntimeClient(config, client=http_client).validate_model(
            "minimaxai/minimax-m3"
        )

        self.assertTrue(result["success"])
        self.assertEqual(payloads[0]["model"], "minimaxai/minimax-m3")
        self.assertEqual(payloads[0]["max_tokens"], 512)
        self.assertEqual(payloads[0]["temperature"], 1.0)
        self.assertEqual(payloads[0]["top_p"], 0.95)
        self.assertFalse(payloads[0]["stream"])

    def test_validate_model_supports_deepseek_payload_profile(self):
        payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/health/ready":
                return httpx.Response(404)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={
                    "data": [
                        {
                            "id": "deepseek-ai/deepseek-v4-flash"
                        }
                    ]
                })
            if request.url.path == "/v1/chat/completions":
                payloads.append(json_from_request(request))
                return httpx.Response(200, json={
                    "choices": [
                        {
                            "message": {
                                "content": "Validation completed.",
                                "reasoning": "Short reasoning trace."
                            }
                        }
                    ],
                    "usage": {
                        "completion_tokens": 10
                    }
                })
            return httpx.Response(500)

        http_client = httpx.Client(
            base_url="https://integrate.api.nvidia.com/v1",
            transport=httpx.MockTransport(handler),
        )
        config = NIMRuntimeConfig(
            base_url="https://integrate.api.nvidia.com/v1",
            model_profiles=self.payload_profiles(),
        )

        result = NIMRuntimeClient(config, client=http_client).validate_model(
            "deepseek-ai/deepseek-v4-flash"
        )

        self.assertTrue(result["success"])
        self.assertEqual(payloads[0]["model"], "deepseek-ai/deepseek-v4-flash")
        self.assertEqual(payloads[0]["max_tokens"], 512)
        self.assertEqual(payloads[0]["temperature"], 1.0)
        self.assertEqual(payloads[0]["top_p"], 0.95)
        self.assertEqual(
            payloads[0]["chat_template_kwargs"],
            {
                "thinking": True,
                "reasoning_effort": "high",
            },
        )

    def test_validate_model_supports_glm_streaming_payload_profile(self):
        payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/health/ready":
                return httpx.Response(404)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={
                    "data": [
                        {
                            "id": "z-ai/glm-5.1"
                        }
                    ]
                })
            if request.url.path == "/v1/chat/completions":
                payloads.append(json_from_request(request))
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=(
                        'data: {"choices":[{"delta":{"content":"Validation"}}]}\n\n'
                        'data: {"choices":[{"delta":{"content":" completed."}}],'
                        '"usage":{"completion_tokens":2}}\n\n'
                        "data: [DONE]\n\n"
                    ),
                )
            return httpx.Response(500)

        http_client = httpx.Client(
            base_url="https://integrate.api.nvidia.com/v1",
            transport=httpx.MockTransport(handler),
        )
        config = NIMRuntimeConfig(
            base_url="https://integrate.api.nvidia.com/v1",
            model_profiles=self.payload_profiles(),
        )

        result = NIMRuntimeClient(config, client=http_client).validate_model("z-ai/glm-5.1")

        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"]["output_tokens"], 2)
        self.assertEqual(payloads[0]["model"], "z-ai/glm-5.1")
        self.assertEqual(payloads[0]["max_tokens"], 16384)
        self.assertEqual(payloads[0]["temperature"], 1.0)
        self.assertEqual(payloads[0]["top_p"], 1.0)
        self.assertTrue(payloads[0]["stream"])

    def test_validate_model_supports_llama_payload_profile(self):
        payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/health/ready":
                return httpx.Response(404)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={
                    "data": [
                        {
                            "id": "meta/llama-3.1-8b-instruct"
                        }
                    ]
                })
            if request.url.path == "/v1/chat/completions":
                payloads.append(json_from_request(request))
                return httpx.Response(200, json={
                    "choices": [
                        {
                            "message": {
                                "content": "Validation completed."
                            }
                        }
                    ],
                    "usage": {
                        "completion_tokens": 8
                    }
                })
            return httpx.Response(500)

        http_client = httpx.Client(
            base_url="https://integrate.api.nvidia.com/v1",
            transport=httpx.MockTransport(handler),
        )
        config = NIMRuntimeConfig(
            base_url="https://integrate.api.nvidia.com/v1",
            model_profiles=self.payload_profiles(),
        )

        result = NIMRuntimeClient(config, client=http_client).validate_model(
            "meta/llama-3.1-8b-instruct"
        )

        self.assertTrue(result["success"])
        self.assertEqual(payloads[0]["model"], "meta/llama-3.1-8b-instruct")
        self.assertEqual(payloads[0]["max_tokens"], 1024)
        self.assertEqual(payloads[0]["temperature"], 0.2)
        self.assertEqual(payloads[0]["top_p"], 0.7)
        self.assertFalse(payloads[0]["stream"])

    def test_validate_model_supports_minimax_m2_payload_profile(self):
        payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/health/ready":
                return httpx.Response(404)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={
                    "data": [
                        {
                            "id": "minimaxai/minimax-m2.7"
                        }
                    ]
                })
            if request.url.path == "/v1/chat/completions":
                payloads.append(json_from_request(request))
                return httpx.Response(200, json={
                    "choices": [
                        {
                            "message": {
                                "content": "Validation completed."
                            }
                        }
                    ],
                    "usage": {
                        "completion_tokens": 8
                    }
                })
            return httpx.Response(500)

        http_client = httpx.Client(
            base_url="https://integrate.api.nvidia.com/v1",
            transport=httpx.MockTransport(handler),
        )
        config = NIMRuntimeConfig(
            base_url="https://integrate.api.nvidia.com/v1",
            model_profiles=self.payload_profiles(),
        )

        result = NIMRuntimeClient(config, client=http_client).validate_model(
            "minimaxai/minimax-m2.7"
        )

        self.assertTrue(result["success"])
        self.assertEqual(payloads[0]["model"], "minimaxai/minimax-m2.7")
        self.assertEqual(payloads[0]["max_tokens"], 8192)
        self.assertEqual(payloads[0]["temperature"], 1.0)
        self.assertEqual(payloads[0]["top_p"], 0.95)
        self.assertFalse(payloads[0]["stream"])

    def test_validate_model_fails_when_payload_profile_is_missing(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/health/ready":
                return httpx.Response(404)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={
                    "data": [
                        {
                            "id": "unknown-model"
                        }
                    ]
                })
            return httpx.Response(500)

        http_client = httpx.Client(
            base_url="https://integrate.api.nvidia.com/v1",
            transport=httpx.MockTransport(handler),
        )
        config = NIMRuntimeConfig(
            base_url="https://integrate.api.nvidia.com/v1",
            model_profiles=self.payload_profiles(),
        )

        result = NIMRuntimeClient(config, client=http_client).validate_model("unknown-model")

        self.assertFalse(result["success"])
        self.assertIn("no model payload profile configured", result["error_message"])
        self.assertIn("configured_models", result["nim"])

    def test_config_from_env_uses_nvidia_api_key_and_model_profiles(self):
        with patch.dict(
            "os.environ",
            {
                "NIM_BASE_URL": "https://integrate.api.nvidia.com/v1",
                "NVIDIA_API_KEY": "nvapi-test",
            },
            clear=True,
        ):
            config = NIMRuntimeConfig.from_env()

        self.assertEqual(config.api_key, "nvapi-test")
        self.assertIn("minimaxai/minimax-m3", config.model_profiles)
        self.assertIn("deepseek-ai/deepseek-v4-flash", config.model_profiles)
        self.assertIn("z-ai/glm-5.1", config.model_profiles)
        self.assertIn("meta/llama-3.1-8b-instruct", config.model_profiles)
        self.assertIn("minimaxai/minimax-m2.7", config.model_profiles)
        self.assertEqual(
            config.model_profiles["minimaxai/minimax-m3"].max_tokens,
            512,
        )

    def test_config_from_env_requires_base_url(self):
        with patch.dict("os.environ", {"NIM_BASE_URL": ""}, clear=True):
            with self.assertRaisesRegex(ValueError, "NIM_BASE_URL is required"):
                NIMRuntimeConfig.from_env()


if __name__ == "__main__":
    unittest.main()
