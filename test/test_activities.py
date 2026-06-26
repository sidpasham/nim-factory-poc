import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from activities import _failure_stage_and_reason  # noqa: E402


class ActivitiesMetricsTests(unittest.TestCase):
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
