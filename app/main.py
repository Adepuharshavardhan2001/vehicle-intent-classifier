from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.schemas import PredictRequest, PredictResponse, LogResponse, HealthCheck
from app.model import Classifier
from app.crud import create_log, get_recent_logs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    init_db()
    logger.info("Application ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="In-Vehicle Intent Classifier",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    classifier = Classifier()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    classifier = None


@app.get("/")
def read_root():
    return {"message": "In-Vehicle Classifier is running!"}


@app.get("/health", response_model=HealthCheck)
def health_check(db: Session = Depends(get_db)):
    health = {"status": "healthy", "checks": {}}

    try:
        db.execute(text("SELECT 1"))
        health["checks"]["database"] = "ok"
    except Exception:
        health["checks"]["database"] = "unhealthy"
        health["status"] = "degraded"

    if classifier is None or not classifier.is_ready():
        health["checks"]["model"] = "unhealthy"
        health["status"] = "degraded"
    else:
        health["checks"]["model"] = "ok"

    return health


@app.get("/ready")
def readiness(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        if classifier is None or not classifier.is_ready():
            raise HTTPException(status_code=503, detail="Model not ready")
        return {"status": "ready"}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Not ready")


@app.get("/live")
def liveness():
    return {"status": "alive"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        intent, confidence = classifier.predict(request.text)
        create_log(db, request.text, intent, confidence)
        logger.info(f"Predicted intent={intent} confidence={confidence:.2f}")
        return PredictResponse(intent=intent, confidence=confidence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction service unavailable")


@app.get("/logs", response_model=list[LogResponse])
def read_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return get_recent_logs(db, skip=skip, limit=limit)