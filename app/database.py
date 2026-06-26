from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
from typing import Generator
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:password@db:3306/vehicle_logs"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    command_text = Column(String(1000), nullable=False)
    predicted_intent = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<PredictionLog(id={self.id}, intent='{self.predicted_intent}', confidence={self.confidence:.2f})>"


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)