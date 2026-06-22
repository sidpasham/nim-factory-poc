from enum import Enum

from pydantic import BaseModel, Field, field_validator


class PrecisionMode(str, Enum):
    FP16 = "FP16"
    INT8 = "INT8"
    INT4 = "INT4"

    @classmethod
    def normalize(cls, value: object) -> "PrecisionMode":
        normalized = str(value or cls.FP16.value).upper().replace("-", "_").strip()
        aliases = {
            "HALF": cls.FP16.value,
            "8BIT": cls.INT8.value,
            "4BIT": cls.INT4.value,
            "AWQ": cls.INT4.value,
            "INT4_AWQ": cls.INT4.value,
        }
        normalized = aliases.get(normalized, normalized)
        return cls(normalized)


class ModelIngestRequest(BaseModel):
    model_name: str = Field(..., min_length=1)
    target_gpu: str = Field(..., min_length=1)
    target_environment: str = Field(default="kubernetes", min_length=1)
    precision_mode: PrecisionMode = PrecisionMode.FP16

    @field_validator("precision_mode", mode="before")
    @classmethod
    def normalize_precision_mode(cls, value: object) -> PrecisionMode:
        return PrecisionMode.normalize(value)
