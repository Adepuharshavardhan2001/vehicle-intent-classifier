import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert isinstance(data["message"], str)


def test_liveness():
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json().get("status") == "alive"


def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_blank_text():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


@patch("model.Classifier.predict")
def test_predict_valid_text(mock_predict):
    mock_predict.return_value = ("navigate", 0.95)
    response = client.post("/predict", json={"text": "navigate to hospital"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "navigate"
    assert data["confidence"] > 0.5


@patch("model.Classifier.predict")
def test_predict_returns_valid_intent(mock_predict):
    mock_predict.return_value = ("play_music", 0.89)
    response = client.post("/predict", json={"text": "play some music"})
    assert response.status_code == 200
    assert response.json()["intent"] in [
        "play_music",
        "navigate",
        "make_call",
        "send_message",
        "set_climate",
    ]


@patch("crud.get_recent_logs")
def test_get_logs_empty(mock_get_logs):
    mock_get_logs.return_value = []
    response = client.get("/logs")
    assert response.status_code == 200
    assert response.json() == []


@patch("crud.get_recent_logs")
def test_get_logs_with_data(mock_get_logs):
    mock_get_logs.return_value = [
        {
            "id": 1,
            "command_text": "play music",
            "predicted_intent": "play_music",
            "confidence": 0.95,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]
    response = client.get("/logs")
    assert response.status_code == 200
    assert len(response.json()) == 1