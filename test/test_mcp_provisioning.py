import os
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
from validation_matrix import VRAMInsufficientError  # noqa: E402


class MCPProvisioningServerTests(unittest.TestCase):
    def test_query_hardware_topology_loads_supported_demo_profiles(self):
        cases = [
            ("NVIDIA GB200", "NVIDIA GB200", "NVLink", "Gen5", True, "nvidia_datacenter_blackwell"),
            ("NVIDIA GB300", "NVIDIA GB300", "NVLink", "Gen5", True, "nvidia_datacenter_blackwell"),
            ("NVIDIA B200", "NVIDIA B200", "NVLink", "Gen5", True, "nvidia_datacenter_blackwell"),
            ("AMD MI355X", "AMD MI355X", "Infinity Fabric", "Gen5", False, "rocm_datacenter"),
            ("AMD MI455X", "AMD MI455X", "Infinity Fabric", "Gen5", False, "rocm_datacenter"),
            ("A10G-24GB", "NVIDIA A10G-24GB", "PCIe", "Gen4", False, "nvidia_datacenter_ampere"),
            ("T4-16GB", "NVIDIA T4-16GB", "PCIe", "Gen3", False, "nvidia_datacenter_turing"),
        ]

        for target_gpu, accelerator, interconnect, pcie_generation, smartnic_active, driver in cases:
            with self.subTest(target_gpu=target_gpu):
                topology = MCPProvisioningServer.query_hardware_topology(target_gpu)

                self.assertEqual(topology["accelerator"], accelerator)
                self.assertEqual(topology["interconnect"], interconnect)
                self.assertEqual(topology["pcie_generation"], pcie_generation)
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
        self.assertEqual(target["hosted_validation_mode"], "nvidia_hosted_external")

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

        with patch.dict("os.environ", {"LLM_GPU_BENCHMARKING_VALIDATION_MODE": "hosted"}, clear=True):
            result = MCPProvisioningServer.run_hardware_test_harness(
                topology,
                "meta/llama-test",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "nvidia_hosted_nim")
        self.assertEqual(result["target_topology"], topology)
        mock_client.validate_model.assert_called_once_with("meta/llama-test")

    @patch("mcp_provisioning.NIMRuntimeClient")
    def test_local_validation_requires_model_name(self, mock_client_class):
        with patch.dict("os.environ", {"LLM_GPU_BENCHMARKING_VALIDATION_MODE": "local"}, clear=True):
            result = MCPProvisioningServer.run_hardware_test_harness(
                {"interconnect": "NVIDIA Hosted API"},
                "",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["backend"], "local_llama_cpp")
        self.assertIn("model_name is required", result["error_message"])
        mock_client_class.from_env.assert_not_called()

    @patch("mcp_provisioning.LocalLLMBenchmarkRunner")
    def test_local_validation_is_default_backend(self, mock_runner_class):
        topology = MCPProvisioningServer.query_hardware_topology("A10G-24GB")
        mock_runner = mock_runner_class.from_env.return_value
        mock_runner.benchmark_model.return_value = {
            "success": True,
            "backend": "local_llama_cpp",
            "metrics": {
                "tokens_per_second": 42,
                "error_rate": 0.0,
                "latency_ms": 3000,
                "output_tokens": 128,
            },
            "target_topology": topology,
        }

        with patch.dict("os.environ", {}, clear=True):
            result = MCPProvisioningServer.run_hardware_test_harness(
                topology,
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                target_gpu="A10G-24GB",
                precision_mode="INT4",
                compile_result={"backend": "local_llama_cpp"},
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "local_llama_cpp")
        mock_runner.benchmark_model.assert_called_once()

    @patch("mcp_provisioning.LocalLLMBenchmarkRunner")
    def test_local_compile_prepares_real_model_artifact(self, mock_runner_class):
        topology = MCPProvisioningServer.query_hardware_topology("A10G-24GB")
        mock_runner = mock_runner_class.from_env.return_value
        mock_runner.prepare_model.return_value = {
            "success": True,
            "backend": "local_llama_cpp",
            "stage": "compile",
            "reason": "local_model_artifact_ready",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "target_gpu": "A10G-24GB",
            "required_vram_gb": 1.1,
            "available_vram_gb": 24,
            "vram_utilization_ratio": 0.0458,
            "precision_mode": "INT4",
        }

        with patch.dict("os.environ", {}, clear=True):
            result = MCPProvisioningServer.evaluate_hardware_fit(
                topology=topology,
                model_name="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                target_gpu="A10G-24GB",
                precision_mode="INT4",
            )

        self.assertEqual(result["backend"], "local_llama_cpp")
        mock_runner.prepare_model.assert_called_once()

    @patch("mcp_provisioning.NIMRuntimeClient")
    def test_validation_reports_runtime_client_error(self, mock_client_class):
        mock_client_class.from_env.side_effect = ValueError("NIM_BASE_URL is required")

        with patch.dict("os.environ", {"LLM_GPU_BENCHMARKING_VALIDATION_MODE": "hosted"}, clear=True):
            result = MCPProvisioningServer.run_hardware_test_harness(
                {"interconnect": "NVIDIA Hosted API"},
                "meta/llama-test",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["backend"], "nvidia_hosted_nim")
        self.assertIn("remote NIM validation failed", result["error_message"])

    def test_validation_matrix_fails_when_model_does_not_fit_target_vram(self):
        topology = MCPProvisioningServer.query_hardware_topology("A10G-24GB")

        with patch.dict("os.environ", {"LLM_GPU_BENCHMARKING_VALIDATION_MODE": "matrix"}, clear=True), \
                self.assertRaises(VRAMInsufficientError) as error:
            MCPProvisioningServer.evaluate_hardware_fit(
                topology=topology,
                model_name="Llama-3-70B",
                target_gpu="A10G-24GB",
                precision_mode="FP16",
            )

        self.assertEqual(error.exception.code, "VRAM_INSUFFICIENT_EXCEPTION")
        self.assertGreater(error.exception.required_vram_gb, error.exception.available_vram_gb)

    def test_validation_matrix_precision_mode_reduces_vram_and_improves_tps(self):
        topology = MCPProvisioningServer.query_hardware_topology("H100-80GB")
        with patch.dict("os.environ", {"LLM_GPU_BENCHMARKING_VALIDATION_MODE": "matrix"}, clear=True):
            fp16_compile = MCPProvisioningServer.evaluate_hardware_fit(
                topology=topology,
                model_name="Llama-3-30B",
                target_gpu="H100-80GB",
                precision_mode="FP16",
            )
            int4_compile = MCPProvisioningServer.evaluate_hardware_fit(
                topology=topology,
                model_name="Llama-3-30B",
                target_gpu="H100-80GB",
                precision_mode="INT4",
            )

            fp16_benchmark = MCPProvisioningServer.run_hardware_test_harness(
                topology,
                "Llama-3-30B",
                target_gpu="H100-80GB",
                precision_mode="FP16",
                compile_result=fp16_compile,
            )
            int4_benchmark = MCPProvisioningServer.run_hardware_test_harness(
                topology,
                "Llama-3-30B",
                target_gpu="H100-80GB",
                precision_mode="INT4",
                compile_result=int4_compile,
            )

        self.assertLess(
            int4_compile["required_vram_gb"],
            fp16_compile["required_vram_gb"],
        )
        self.assertGreater(
            int4_benchmark["metrics"]["tokens_per_second"],
            fp16_benchmark["metrics"]["tokens_per_second"],
        )
        self.assertLess(
            int4_benchmark["metrics"]["accuracy_score"],
            fp16_benchmark["metrics"]["accuracy_score"],
        )


if __name__ == "__main__":
    unittest.main()
