from pydantic import BaseModel
from datetime import datetime

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    intent: str
    confidence: float

class LogResponse(BaseModel):
    id: int
    command_text: str
    predicted_intent: str
    confidence: float
    timestamp: datetime

    class Config:
        from_attributes = True