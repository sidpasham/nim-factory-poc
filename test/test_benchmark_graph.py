import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_graph import (  # noqa: E402
    handle_failure,
    llm_gpu_benchmarking_graph_pipeline,
    route_validation_results,
)


class BenchmarkGraphTests(unittest.TestCase):
    def test_route_validation_results_publishes_successful_high_throughput_run(self):
        with patch.dict("os.environ", {}, clear=True):
            route = route_validation_results({
                "validation_results": {
                    "success": True,
                    "metrics": {
                        "tokens_per_second": 2500
                    }
                }
            })

        self.assertEqual(route, "publish")

    def test_route_validation_results_fails_low_throughput_run(self):
        with patch.dict("os.environ", {"LLM_GPU_BENCHMARKING_MINIMUM_TPS_THRESHOLD": "1200"}, clear=True):
            route = route_validation_results({
                "validation_results": {
                    "success": True,
                    "metrics": {
                        "tokens_per_second": 1200
                    }
                }
            })

        self.assertEqual(route, "fail")

    def test_route_validation_results_uses_configured_tps_threshold(self):
        with patch.dict("os.environ", {"LLM_GPU_BENCHMARKING_MINIMUM_TPS_THRESHOLD": "100"}, clear=True):
            route = route_validation_results({
                "validation_results": {
                    "success": True,
                    "metrics": {
                        "tokens_per_second": 145.97
                    }
                }
            })

        self.assertEqual(route, "publish")

    def test_route_validation_results_fails_unsuccessful_validation_run(self):
        with patch.dict("os.environ", {}, clear=True):
            route = route_validation_results({
                "validation_results": {
                    "success": False,
                    "metrics": {
                        "tokens_per_second": 2800
                    }
                }
            })

        self.assertEqual(route, "fail")

    def test_handle_failure_uses_state_to_build_contextual_error_message(self):
        with patch.dict("os.environ", {"LLM_GPU_BENCHMARKING_MINIMUM_TPS_THRESHOLD": "100"}, clear=True), \
                patch("builtins.print"):
            result = handle_failure({
                "model_name": "llama-demo",
                "target_gpu": "GENERIC-GPU",
                "target_environment": "kubernetes",
                "hardware_topology": {
                    "interconnect": "PCIe"
                },
                "deployment_target": {
                    "environment": "kubernetes"
                },
                "validation_results": {
                    "success": True,
                    "metrics": {
                        "tokens_per_second": 1100,
                        "error_rate": 0.0
                    }
                },
                "status": "Validation_Completed",
                "error_message": ""
            })

        self.assertEqual(result["status"], "Failed")
        self.assertIn("llama-demo", result["error_message"])
        self.assertIn("GENERIC-GPU", result["error_message"])
        self.assertIn("kubernetes", result["error_message"])
        self.assertIn("1100 TPS", result["error_message"])
        self.assertIn("100 TPS threshold", result["error_message"])
        self.assertIn("PCIe", result["error_message"])
        self.assertIn("error_rate=0.0", result["error_message"])

    def test_pipeline_records_precision_profile_for_int4(self):
        with patch.dict("os.environ", {"LLM_GPU_BENCHMARKING_MINIMUM_TPS_THRESHOLD": "1"}, clear=True):
            result = llm_gpu_benchmarking_graph_pipeline.invoke({
                "model_name": "Llama-3-30B",
                "target_gpu": "H100-80GB",
                "target_environment": "kubernetes",
                "precision_mode": "INT4",
                "hardware_topology": {},
                "deployment_target": {},
                "precision_result": {},
                "compile_result": {},
                "validation_results": {},
                "stage_durations": {},
                "status": "Ingested",
                "error_message": "",
            })

        self.assertEqual(result["status"], "Model_Service_Ready_To_Deploy")
        self.assertTrue(result["precision_result"]["profiled"])
        self.assertIn("precision_profile", result["stage_durations"])
        self.assertLess(
            result["validation_results"]["metrics"]["required_vram_gb"],
            result["compile_result"]["available_vram_gb"],
        )

    def test_pipeline_fails_vram_insufficient_compile(self):
        with patch.dict("os.environ", {"LLM_GPU_BENCHMARKING_MINIMUM_TPS_THRESHOLD": "1"}, clear=True):
            result = llm_gpu_benchmarking_graph_pipeline.invoke({
                "model_name": "Llama-3-70B",
                "target_gpu": "A10G-24GB",
                "target_environment": "kubernetes",
                "precision_mode": "FP16",
                "hardware_topology": {},
                "deployment_target": {},
                "precision_result": {},
                "compile_result": {},
                "validation_results": {},
                "stage_durations": {},
                "status": "Ingested",
                "error_message": "",
            })

        self.assertEqual(result["status"], "Failed")
        self.assertEqual(
            result["compile_result"]["error_code"],
            "VRAM_INSUFFICIENT_EXCEPTION",
        )
        self.assertIn("VRAM_INSUFFICIENT_EXCEPTION", result["error_message"])


if __name__ == "__main__":
    unittest.main()
