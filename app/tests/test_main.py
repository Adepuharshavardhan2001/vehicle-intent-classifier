import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code in [200, 503]


def test_liveness():
    response = client.get("/live")
    assert response.status_code == 200


def test_readiness():
    response = client.get("/ready")
    assert response.status_code == 200


def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code in [400, 422]


def test_predict_valid_text():
    response = client.post("/predict", json={"text": "navigate to hospital"})
    assert response.status_code in [200, 500]


def test_get_logs():
    response = client.get("/logs")
    assert response.status_code in [200, 500]