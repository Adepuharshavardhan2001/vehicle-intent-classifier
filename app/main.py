from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine, Base
from app.schemas import PredictRequest, PredictResponse, LogResponse
from app.model import Classifier
from app.crud import create_log, get_recent_logs

# Create Database Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="In-Vehicle Intent Classifier")

# Load the AI model once when the server starts
classifier = Classifier()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "In-Vehicle Classifier is running!"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, db: Session = Depends(get_db)):
    # 1. Run the AI model
    intent, confidence = classifier.predict(request.text)
    
    # 2. Save the prediction to MySQL
    create_log(db, request.text, intent, confidence)
    
    # 3. Return the result to the user
    return PredictResponse(intent=intent, confidence=confidence)

@app.get("/logs", response_model=list[LogResponse])
async def read_logs(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    logs = get_recent_logs(db, skip=skip, limit=limit)
    return logs