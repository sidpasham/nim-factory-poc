import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from schemas import ModelIngestRequest, PrecisionMode  # noqa: E402


class SchemaTests(unittest.TestCase):
    def test_precision_mode_accepts_validation_matrix_modes(self):
        for mode in ("FP16", "INT8", "INT4"):
            with self.subTest(mode=mode):
                request = ModelIngestRequest(
                    model_name="Llama-3-8B",
                    target_gpu="H100-80GB",
                    precision_mode=mode,
                )
                self.assertEqual(request.precision_mode, PrecisionMode(mode))

    def test_precision_mode_rejects_unsupported_modes(self):
        with self.assertRaises(ValidationError):
            ModelIngestRequest(
                model_name="Llama-3-8B",
                target_gpu="H100-80GB",
                precision_mode="FP8",
            )


if __name__ == "__main__":
    unittest.main()
