from dotenv import load_dotenv
load_dotenv()

import os
import time
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is required. "
        "Example: mysql+pymysql://user:pass@host:3306/dbname"
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    command_text = Column(String(1000), nullable=False)
    predicted_intent = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def __repr__(self):
        return (
            f"<PredictionLog(id={self.id}, "
            f"intent='{self.predicted_intent}', "
            f"confidence={self.confidence:.2f})>"
        )


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(retries: int = 5, delay: int = 2) -> None:
    """Create tables. Retries if MySQL isn't ready yet."""
    for attempt in range(1, retries + 1):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except OperationalError:
            if attempt == retries:
                raise
            time.sleep(delay)