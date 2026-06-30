import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from activities import _build_pipeline_summary, _failure_stage_and_reason  # noqa: E402


class ActivitiesMetricsTests(unittest.TestCase):
    def test_pipeline_summary_identifies_deployable_target(self):
        summary = _build_pipeline_summary(
            {
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "target_gpu": "A10G-24GB",
                "target_environment": "kubernetes",
                "precision_mode": "INT4",
                "hardware_topology": {
                    "accelerator": "NVIDIA A10G-24GB",
                },
                "deployment_target": {
                    "environment": "kubernetes",
                },
                "precision_result": {},
                "compile_result": {},
                "stage_durations": {},
                "error_message": "",
            },
            {"tokens_per_second": 145.97},
            "Model_Service_Ready_To_Deploy",
        )

        self.assertTrue(summary["deployable"])
        self.assertEqual(summary["hardware"], "A10G-24GB")
        self.assertEqual(summary["target_gpu"], "A10G-24GB")
        self.assertEqual(
            summary["deployable_on"],
            {
                "target_gpu": "A10G-24GB",
                "target_environment": "kubernetes",
                "precision_mode": "INT4",
            },
        )

    def test_pipeline_summary_does_not_set_deployable_on_for_failures(self):
        summary = _build_pipeline_summary(
            {
                "model_name": "Llama-3-70B",
                "target_gpu": "A10G-24GB",
                "target_environment": "kubernetes",
                "precision_mode": "FP16",
                "hardware_topology": {},
                "deployment_target": {},
                "precision_result": {},
                "compile_result": {},
                "stage_durations": {},
                "error_message": "VRAM is insufficient.",
            },
            {"tokens_per_second": 0},
            "Failed",
        )

        self.assertFalse(summary["deployable"])
        self.assertIsNone(summary["deployable_on"])
        self.assertEqual(
            summary["evaluated_target"],
            {
                "target_gpu": "A10G-24GB",
                "target_environment": "kubernetes",
                "precision_mode": "FP16",
            },
        )

    def test_failure_reason_prefers_compile_error_for_compile_failure(self):
        stage, reason = _failure_stage_and_reason({
            "compile_result": {
                "success": False,
                "stage": "compile",
                "reason": "target_vram_insufficient",
                "error_code": "VRAM_INSUFFICIENT_EXCEPTION",
            },
            "validation_results": {},
        })

        self.assertEqual(stage, "compile")
        self.assertEqual(reason, "VRAM_INSUFFICIENT_EXCEPTION")

    def test_failure_reason_marks_low_throughput_as_benchmark_failure(self):
        with patch.dict(
            "os.environ",
            {"LLM_GPU_BENCHMARKING_MINIMUM_TPS_THRESHOLD": "100"},
            clear=True,
        ):
            stage, reason = _failure_stage_and_reason({
                "compile_result": {
                    "success": True,
                    "stage": "compile",
                    "reason": "local_model_artifact_ready",
                },
                "validation_results": {
                    "success": True,
                    "metrics": {
                        "tokens_per_second": 46.63,
                    },
                },
            })

        self.assertEqual(stage, "benchmark")
        self.assertEqual(reason, "throughput_below_threshold")

    def test_failure_reason_uses_validation_error_for_harness_failure(self):
        stage, reason = _failure_stage_and_reason({
            "compile_result": {
                "success": True,
                "stage": "compile",
                "reason": "local_model_artifact_ready",
            },
            "validation_results": {
                "success": False,
                "stage": "benchmark",
                "error_code": "BENCHMARK_TIMEOUT",
            },
        })

        self.assertEqual(stage, "benchmark")
        self.assertEqual(reason, "BENCHMARK_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
