from sqlalchemy.orm import Session
from app.database import PredictionLog

def create_log(db: Session, text: str, intent: str, confidence: float):
    db_log = PredictionLog(
        command_text=text,
        predicted_intent=intent,
        confidence=confidence
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_recent_logs(db: Session, skip: int = 0, limit: int = 10):
    return db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).offset(skip).limit(limit).all()