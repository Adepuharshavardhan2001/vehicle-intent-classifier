from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List
import logging
from app.database import PredictionLog

logger = logging.getLogger(__name__)


def create_log(db: Session, text: str, intent: str, confidence: float) -> PredictionLog:
    if not text or not text.strip():
        raise ValueError("text cannot be empty")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0")

    try:
        db_log = PredictionLog(
            command_text=text,
            predicted_intent=intent,
            confidence=confidence
        )
        db.add(db_log)
        db.commit()
        db.refresh(db_log)
        logger.info(f"Prediction logged: '{text[:30]}...' → {intent} ({confidence:.2f})")
        return db_log
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to create log: {e}")
        raise


def get_recent_logs(db: Session, skip: int = 0, limit: int = 10) -> List[PredictionLog]:
    try:
        return (
            db.query(PredictionLog)
            .order_by(PredictionLog.timestamp.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
    except SQLAlchemyError as e:
        logger.error(f"Failed to fetch logs: {e}")
        raise


def get_total_count(db: Session) -> int:
    try:
        return db.query(PredictionLog).count()
    except SQLAlchemyError as e:
        logger.error(f"Failed to count logs: {e}")
        return 0