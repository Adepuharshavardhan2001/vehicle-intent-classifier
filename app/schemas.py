from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Voice command text to classify",
        examples=["navigate to the nearest hospital"],
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text cannot be blank")
        return v


class PredictResponse(BaseModel):
    intent: str = Field(..., description="Predicted intent class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class LogResponse(BaseModel):
    id: int = Field(..., description="Log entry ID")
    command_text: str = Field(..., description="Input voice command")
    predicted_intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    timestamp: datetime = Field(..., description="UTC timestamp")

    model_config = ConfigDict(from_attributes=True)


class HealthCheck(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    checks: dict[str, Literal["ok", "unhealthy"]]


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")