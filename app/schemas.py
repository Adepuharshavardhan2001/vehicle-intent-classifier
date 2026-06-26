from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List, Optional


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Voice command text to classify",
        examples=["navigate to the nearest hospital"]
    )


class PredictResponse(BaseModel):
    intent: str = Field(..., description="Predicted intent class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class LogResponse(BaseModel):
    id: int
    command_text: str
    predicted_intent: str
    confidence: float
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class HealthCheck(BaseModel):
    status: str
    checks: dict


class ErrorResponse(BaseModel):
    detail: str