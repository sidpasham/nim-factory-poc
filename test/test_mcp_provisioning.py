import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from mcp_provisioning import MCPProvisioningServer  # noqa: E402


class MCPProvisioningServerTests(unittest.TestCase):
    def test_query_hardware_topology_loads_matching_nvlink_profile(self):
        topology = MCPProvisioningServer.query_hardware_topology("NVIDIA-GB200")

        self.assertEqual(topology["interconnect"], "NVLink")
        self.assertEqual(topology["pcie_generation"], "Gen5")
        self.assertTrue(topology["smartnic_active"])
        self.assertEqual(topology["optimized_driver"], "v550_prod")

    def test_query_hardware_topology_uses_default_profile(self):
        topology = MCPProvisioningServer.query_hardware_topology("GENERIC-GPU")

        self.assertEqual(topology["interconnect"], "PCIe")
        self.assertEqual(topology["pcie_generation"], "Gen4")
        self.assertFalse(topology["smartnic_active"])
        self.assertEqual(topology["optimized_driver"], "v540_generic")

    @patch("mcp_provisioning.random.uniform", return_value=18.5)
    @patch("mcp_provisioning.random.randint", return_value=2750)
    @patch("mcp_provisioning.random.random", return_value=0.10)
    def test_validation_harness_uses_interconnect_profile(
        self,
        mock_random,
        mock_randint,
        mock_uniform
    ):
        result = MCPProvisioningServer.run_hardware_test_harness({"interconnect": "NVLink"})

        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"]["tokens_per_second"], 2750)
        self.assertEqual(result["metrics"]["error_rate"], 0.0)
        self.assertEqual(result["metrics"]["latency_ms"], 18.5)
        mock_random.assert_called_once_with()
        mock_randint.assert_called_once_with(2500, 3000)
        mock_uniform.assert_called_once_with(12.5, 24.0)

    @patch("mcp_provisioning.random.uniform", return_value=20.0)
    @patch("mcp_provisioning.random.randint", return_value=1100)
    @patch("mcp_provisioning.random.random", return_value=0.99)
    def test_validation_harness_reports_configured_failure_error_rate(
        self,
        _mock_random,
        _mock_randint,
        _mock_uniform
    ):
        result = MCPProvisioningServer.run_hardware_test_harness({"interconnect": "PCIe"})

        self.assertFalse(result["success"])
        self.assertEqual(result["metrics"]["tokens_per_second"], 1100)
        self.assertEqual(result["metrics"]["error_rate"], 0.12)


if __name__ == "__main__":
    unittest.main()
