import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import certifi


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from local_llm_runtime import (  # noqa: E402
    LocalLLMBenchmarkRunner,
    LocalLLMRuntimeConfig,
)


def runtime_config(root_dir: Path, profiles_path: Path) -> LocalLLMRuntimeConfig:
    return LocalLLMRuntimeConfig(
        root_dir=root_dir,
        profiles_path=profiles_path,
        repo_url="https://github.com/ggml-org/llama.cpp.git",
        repo_ref="test-ref",
        cmake_args=("-DGGML_METAL=OFF",),
        build_jobs=1,
        bench_repetitions=1,
        explicit_cli_path=None,
        ca_bundle_path=Path(certifi.where()),
        download_timeout_seconds=30,
        benchmark_timeout_seconds=30,
        warmup_on_startup=True,
        warmup_models=("all",),
        warmup_precision_modes=("INT4",),
        warmup_required=True,
    )


def write_profiles(path: Path) -> None:
    path.write_text(json.dumps({
        "profiles": [
            {
                "model_name": "demo/local-model",
                "aliases": ["demo"],
                "hf_repo": "demo/local-model",
                "architecture": "demo",
                "license": "test",
                "parameter_count_billion": 0.5,
                "context_size": 512,
                "max_tokens": 16,
                "temperature": 0.0,
                "benchmark_prompt": "Say hello.",
                "artifacts": {
                    "INT4": {
                        "filename": "demo-q4.gguf",
                        "quantization": "Q4_K_M"
                    },
                    "INT8": {
                        "filename": "demo-q8.gguf",
                        "quantization": "Q8_0"
                    }
                }
            }
        ]
    }), encoding="utf-8")


class LocalLLMRuntimeTests(unittest.TestCase):
    def test_llama_cpp_cache_validates_against_manifest_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profiles_path = root / "profiles.json"
            write_profiles(profiles_path)
            runner = LocalLLMBenchmarkRunner(runtime_config(root, profiles_path))
            cli_path = root / "llama.cpp" / "build" / "bin" / "llama-cli"
            bench_path = root / "llama.cpp" / "build" / "bin" / "llama-bench"
            cli_path.parent.mkdir(parents=True)
            cli_path.write_text("cli", encoding="utf-8")
            bench_path.write_text("bench", encoding="utf-8")

            self.assertFalse(runner._llama_cpp_cache_valid(cli_path, bench_path))

            runner._write_llama_cpp_cache_manifest(cli_path, bench_path)

            self.assertTrue(runner._llama_cpp_cache_valid(cli_path, bench_path))
            manifest = runner.cache_status()["manifest"]
            self.assertEqual(manifest["llama_cpp"]["repo_ref"], "test-ref")

    def test_warmup_cache_prepares_selected_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profiles_path = root / "profiles.json"
            write_profiles(profiles_path)
            runner = LocalLLMBenchmarkRunner(runtime_config(root, profiles_path))
            cli_path = root / "llama.cpp" / "build" / "bin" / "llama-cli"
            bench_path = root / "llama.cpp" / "build" / "bin" / "llama-bench"
            cli_path.parent.mkdir(parents=True)
            cli_path.write_text("cli", encoding="utf-8")
            bench_path.write_text("bench", encoding="utf-8")
            model_path = root / "models" / "demo-q4.gguf"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"model")

            with patch.object(runner, "ensure_llama_cpp", return_value=cli_path), \
                    patch.object(runner, "ensure_model_artifact", return_value=model_path) as ensure_model:
                result = runner.warmup_cache()

            self.assertTrue(result["success"])
            self.assertEqual(result["models"][0]["model_name"], "demo/local-model")
            self.assertEqual(
                result["models"][0]["artifacts"][0]["precision_mode"],
                "INT4",
            )
            ensure_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
