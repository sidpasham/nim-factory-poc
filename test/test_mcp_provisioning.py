import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from mcp_provisioning import (  # noqa: E402
    MCPProvisioningServer,
    UnsupportedProvisioningTargetError,
)


class MCPProvisioningServerTests(unittest.TestCase):
    def test_query_hardware_topology_loads_supported_demo_profiles(self):
        cases = [
            ("NVIDIA GB200", "NVIDIA GB200", "NVLink", True, "nvidia_datacenter_blackwell"),
            ("NVIDIA GB300", "NVIDIA GB300", "NVLink", True, "nvidia_datacenter_blackwell"),
            ("NVIDIA B200", "NVIDIA B200", "NVLink", True, "nvidia_datacenter_blackwell"),
            ("AMD MI355X", "AMD MI355X", "Infinity Fabric", False, "rocm_datacenter"),
            ("AMD MI455X", "AMD MI455X", "Infinity Fabric", False, "rocm_datacenter"),
        ]

        for target_gpu, accelerator, interconnect, smartnic_active, driver in cases:
            with self.subTest(target_gpu=target_gpu):
                topology = MCPProvisioningServer.query_hardware_topology(target_gpu)

                self.assertEqual(topology["accelerator"], accelerator)
                self.assertEqual(topology["interconnect"], interconnect)
                self.assertEqual(topology["pcie_generation"], "Gen5")
                self.assertEqual(topology["smartnic_active"], smartnic_active)
                self.assertEqual(topology["optimized_driver"], driver)

    def test_query_hardware_topology_rejects_unknown_profile(self):
        with self.assertRaisesRegex(
            UnsupportedProvisioningTargetError,
            "Supported hardware targets",
        ):
            MCPProvisioningServer.query_hardware_topology("GENERIC-GPU")

    def test_query_deployment_target_loads_rancher_kubernetes_profile(self):
        target = MCPProvisioningServer.query_deployment_target("rancher-desktop")

        self.assertEqual(target["environment"], "kubernetes")
        self.assertEqual(target["platform"], "Rancher Desktop k3s")
        self.assertEqual(target["nim_endpoint_mode"], "nvidia_hosted_external")

    def test_query_deployment_target_loads_cloud_profile(self):
        target = MCPProvisioningServer.query_deployment_target("oci")

        self.assertEqual(target["environment"], "cloud")
        self.assertEqual(target["control_plane"], "managed Kubernetes")

    def test_query_deployment_target_rejects_unknown_target(self):
        with self.assertRaisesRegex(
            UnsupportedProvisioningTargetError,
            "Supported deployment targets",
        ):
            MCPProvisioningServer.query_deployment_target("bare-metal")

    @patch("mcp_provisioning.NIMRuntimeClient")
    def test_validation_harness_delegates_to_runtime_client(self, mock_client_class):
        topology = {"interconnect": "NVIDIA Hosted API"}
        mock_client = mock_client_class.from_env.return_value.__enter__.return_value
        mock_client.validate_model.return_value = {
            "success": True,
            "backend": "nvidia_hosted_nim",
            "metrics": {
                "tokens_per_second": 1400,
                "error_rate": 0.0,
                "latency_ms": 120,
            }
        }

        result = MCPProvisioningServer.run_hardware_test_harness(
            topology,
            "meta/llama-test",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "nvidia_hosted_nim")
        self.assertEqual(result["target_topology"], topology)
        mock_client.validate_model.assert_called_once_with("meta/llama-test")

    @patch("mcp_provisioning.NIMRuntimeClient")
    def test_validation_requires_model_name(self, mock_client_class):
        result = MCPProvisioningServer.run_hardware_test_harness(
            {"interconnect": "NVIDIA Hosted API"},
            "",
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["backend"], "nvidia_hosted_nim")
        self.assertIn("model_name is required", result["error_message"])
        mock_client_class.from_env.assert_not_called()

    @patch("mcp_provisioning.NIMRuntimeClient")
    def test_validation_reports_runtime_client_error(self, mock_client_class):
        mock_client_class.from_env.side_effect = ValueError("NIM_BASE_URL is required")

        result = MCPProvisioningServer.run_hardware_test_harness(
            {"interconnect": "NVIDIA Hosted API"},
            "meta/llama-test",
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["backend"], "nvidia_hosted_nim")
        self.assertIn("remote NIM validation failed", result["error_message"])


if __name__ == "__main__":
    unittest.main()
