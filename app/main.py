from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.database import get_db, init_db
from app.schemas import PredictRequest, PredictResponse, LogResponse, HealthCheck
from app.model import Classifier
from app.crud import create_log, get_recent_logs, get_total_count
import logging

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = Classifier()


@app.get("/")
def read_root():
    return {"message": "In-Vehicle Classifier is running!"}


@app.get("/health", response_model=HealthCheck)
async def health_check(db: Session = Depends(get_db)):
    health = {"status": "healthy", "checks": {}}

    try:
        db.execute("SELECT 1")
        health["checks"]["database"] = "ok"
    except Exception:
        health["checks"]["database"] = "unhealthy"
        health["status"] = "degraded"

    health["checks"]["model"] = "ok" if classifier.is_ready() else "unhealthy"
    if health["checks"]["model"] == "unhealthy":
        health["status"] = "degraded"

    return health


@app.get("/ready")
async def readiness():
    return {"status": "ready"}


@app.get("/live")
async def liveness():
    return {"status": "alive"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, db: Session = Depends(get_db)):
    try:
        intent, confidence = classifier.predict(request.text)
        create_log(db, request.text, intent, confidence)
        return PredictResponse(intent=intent, confidence=confidence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction service unavailable")


@app.get("/logs", response_model=list[LogResponse])
async def read_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    return get_recent_logs(db, skip=skip, limit=limit)