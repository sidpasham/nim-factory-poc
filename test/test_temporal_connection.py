import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from temporal_connection import temporal_address_candidates  # noqa: E402


class TemporalConnectionTests(unittest.TestCase):
    def test_temporal_address_candidates_use_configured_address_first(self):
        with patch.dict(
            "os.environ",
            {
                "TEMPORAL_ADDRESS": "temporal.example:7233",
                "LLM_GPU_BENCHMARKING_TEMPORAL_SERVICE_HOST": "10.43.188.43",
                "LLM_GPU_BENCHMARKING_TEMPORAL_SERVICE_PORT_GRPC": "7233",
            },
            clear=True,
        ):
            candidates = temporal_address_candidates("temporal:7233")

        self.assertEqual(
            candidates,
            [
                "temporal.example:7233",
                "10.43.188.43:7233",
                "temporal:7233",
            ],
        )

    def test_temporal_address_candidates_fall_back_to_service_port(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_GPU_BENCHMARKING_TEMPORAL_SERVICE_HOST": "10.43.188.43",
                "LLM_GPU_BENCHMARKING_TEMPORAL_SERVICE_PORT": "7233",
            },
            clear=True,
        ):
            candidates = temporal_address_candidates("temporal:7233")

        self.assertEqual(candidates, ["temporal:7233", "10.43.188.43:7233"])


if __name__ == "__main__":
    unittest.main()
