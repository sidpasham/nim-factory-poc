from enum import Enum

from pydantic import BaseModel, Field


class PrecisionMode(str, Enum):
    FP16 = "FP16"
    INT8 = "INT8"
    INT4 = "INT4"


class ModelIngestRequest(BaseModel):
    model_name: str = Field(..., min_length=1)
    target_gpu: str = Field(..., min_length=1)
    target_environment: str = Field(default="kubernetes", min_length=1)
    precision_mode: PrecisionMode = PrecisionMode.FP16
