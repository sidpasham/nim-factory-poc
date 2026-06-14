import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from factory_graph import handle_failure, route_validation_results  # noqa: E402


class FactoryGraphTests(unittest.TestCase):
    def test_route_validation_results_publishes_successful_high_throughput_run(self):
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
        route = route_validation_results({
            "validation_results": {
                "success": True,
                "metrics": {
                    "tokens_per_second": 1200
                }
            }
        })

        self.assertEqual(route, "fail")

    def test_route_validation_results_fails_unsuccessful_validation_run(self):
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
        with patch("builtins.print"):
            result = handle_failure({
                "model_name": "llama-demo",
                "target_gpu": "GENERIC-GPU",
                "hardware_topology": {
                    "interconnect": "PCIe"
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
        self.assertIn("1100 TPS", result["error_message"])
        self.assertIn("PCIe", result["error_message"])
        self.assertIn("error_rate=0.0", result["error_message"])


if __name__ == "__main__":
    unittest.main()
