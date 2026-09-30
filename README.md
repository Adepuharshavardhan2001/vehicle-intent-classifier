# In-Vehicle Intent Classifier

A production-ready ML microservice that classifies in-vehicle voice commands using a fine-tuned DistilBERT model. Built with FastAPI, MySQL, Docker, and shipped via a full CI/CD pipeline to GitHub Container Registry.

[![CI](https://github.com/Adepuharshavardhan2001/vehicle-intent-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Adepuharshavardhan2001/vehicle-intent-classifier/actions/workflows/ci.yml)
[![CD](https://github.com/Adepuharshavardhan2001/vehicle-intent-classifier/actions/workflows/cd.yml/badge.svg)](https://github.com/Adepuharshavardhan2001/vehicle-intent-classifier/actions/workflows/cd.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)

---

## Overview

Modern vehicles use voice assistants to let drivers control navigation, media, climate, and communications hands-free. The core ML problem: **classify the driver's spoken command into a known intent** before the assistant can act.

This project is a full MLOps implementation of that component:

- A **fine-tuned DistilBERT** model achieving 94% validation accuracy on 5+ intent classes
- A **FastAPI service** with prediction logging to MySQL
- A **multi-stage Docker build** with the model distributed separately via GitHub Releases
- A **complete CI/CD pipeline** that tests, builds, and publishes the image to GitHub Container Registry on every push
- **Kubernetes manifests** for production deployment

---

## Evaluation Results

Fine-tuned on 1,500+ in-vehicle voice commands across 5+ intent categories:

| Metric | Result |
|---|---|
| **Validation accuracy** | **94%** (held-out split) |
| **Base model** | `distilbert-base-uncased` |
| **Training samples** | 1,500+ |
| **Intent classes** | 5+ |
| **Max sequence length** | 128 tokens |

Per-class performance is verified by `app/tests/test_model.py::test_model_accuracy_on_samples`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML Framework** | PyTorch 2.1, HuggingFace Transformers 4.35 |
| **Model** | DistilBERT (fine-tuned) |
| **API** | FastAPI, Uvicorn |
| **Database** | MySQL 8.0, SQLAlchemy 2.0 |
| **Validation** | Pydantic v2 |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Registry** | GitHub Container Registry (GHCR) |
| **Orchestration** | Kubernetes |
| **Testing** | pytest, pytest-cov, httpx |

---

## Project Structure

```
vehicle-intent-classifier/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Tests on every push
│       └── cd.yml              # Build + push to GHCR on main
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── model.py                # DistilBERT wrapper (inference)
│   ├── database.py             # SQLAlchemy setup + connection pooling
│   ├── crud.py                 # DB operations
│   ├── schemas.py              # Pydantic request/response contracts
│   └── tests/
│       ├── __init__.py
│       ├── test_main.py        # API endpoint tests
│       └── test_model.py       # Model inference + accuracy tests
├── k8s/
│   ├── configmap.yaml          # Non-sensitive config
│   ├── secret.yaml.example     # Placeholder for sensitive config
│   ├── pvc.yaml                # Persistent volumes
│   ├── deployment.yaml         # K8s Deployment
│   └── service.yaml            # K8s Service
├── models/
│   ├── distilbert_finetuned/   # (gitignored) model artifacts
│   └── label_map.json          # (gitignored) intent label mapping
├── data/
│   └── dataset.xlsx            # Training dataset (1,500+ commands)
├── train.py                    # Training script
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Local orchestration
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Dev/test dependencies
├── requirements-train.txt      # Training dependencies
├── pyproject.toml              # pytest + ruff config
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Full health check (DB + model) |
| `GET` | `/ready` | Readiness probe for K8s |
| `GET` | `/live` | Liveness probe for K8s |
| `POST` | `/predict` | Classify a voice command |
| `GET` | `/logs` | Recent prediction logs |

### Example — POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "navigate to the nearest hospital"}'
```

Response:

```json
{
  "intent": "navigate",
  "confidence": 0.95
}
```

### Example — GET /logs

```bash
curl "http://localhost:8000/logs?skip=0&limit=10"
```

Response:

```json
[
  {
    "id": 1,
    "command_text": "play some music",
    "predicted_intent": "play_music",
    "confidence": 0.97,
    "timestamp": "2026-09-30T10:00:00Z"
  }
]
```

---

## Quick Start

### Prerequisites

- Python 3.11
- MySQL 8.0 (running on `localhost:3306`)
- The trained model — download from [GitHub Releases v1.0](https://github.com/Adepuharshavardhan2001/vehicle-intent-classifier/releases/tag/v1.0)

### 1. Clone

```bash
git clone https://github.com/Adepuharshavardhan2001/vehicle-intent-classifier.git
cd vehicle-intent-classifier
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the trained model

The model (~236 MB) is distributed via GitHub Releases (not committed to git):

```bash
mkdir -p models
cd models
wget https://github.com/Adepuharshavardhan2001/vehicle-intent-classifier/releases/download/v1.0/distilbert_finetuned.tar.gz
tar -xzf distilbert_finetuned.tar.gz
cd ..
```

Expected structure:

```
models/
├── distilbert_finetuned/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
└── label_map.json
```

### 4. Configure environment

Create `.env`:

```
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/vehicle_logs
MODEL_PATH=models/distilbert_finetuned
LABEL_PATH=models/label_map.json
```

### 5. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open Swagger UI: **http://localhost:8000/docs**

---

## Training

To retrain the model from scratch:

```bash
pip install -r requirements-train.txt
python train.py
```

The script:

- Loads `data/dataset.xlsx`
- Tokenizes with `distilbert-base-uncased` (max 128 tokens)
- Trains for up to 5 epochs with early stopping (patience=2)
- Uses linear warmup + decay LR schedule
- Saves the best checkpoint to `models/distilbert_finetuned/`
- Logs validation accuracy + classification report per epoch
- Saves training metadata to `models/distilbert_finetuned/metadata.json`

---

## Testing

```bash
pytest app/tests/ -v
```

Expected: **9 passed, 11 skipped** (model tests skip if the model isn't downloaded).

All tests use mocks for external services. No real MySQL or model required for the API tests.

Coverage report is generated automatically (configured in `pyproject.toml`).

---

## Docker

### Build

```bash
docker build -t vehicle-intent .
```

The Dockerfile:

- Uses a **multi-stage build** (`builder` + runtime)
- Installs **CPU-only torch** (saves ~600 MB vs CUDA version)
- **Downloads the model** from GitHub Releases during the build
- Runs as **non-root user** (`appuser`)
- Final image size: ~1.2 GB

### Run

```bash
docker run -p 8000:8000 \
  -e DATABASE_URL="mysql+pymysql://root:password@host.docker.internal:3306/vehicle_logs" \
  vehicle-intent
```

### Docker Compose

```bash
docker-compose up -d
```

Starts MySQL + API together. Requires `.env` with:

```
MYSQL_ROOT_PASSWORD=password
MYSQL_DATABASE=vehicle_logs
```

---

## CI/CD Pipeline

### CI (`ci.yml`) — on every push and PR

1. Spins up a MySQL 8.0 service
2. Installs dependencies
3. Runs `pytest app/tests/ -v`
4. Reports status

### CD (`cd.yml`) — on every push to `main`

1. **Job 1: test** — runs full test suite (MySQL service + env vars)
2. **Job 2: build-and-push** (only if tests pass):
   - Sets up Docker Buildx
   - Logs in to GitHub Container Registry
   - Builds the Docker image
   - Pushes to `ghcr.io/adepuharshavardhan2001/vehicle-intent:latest` and `:sha-<commit>`

Uses a **Personal Access Token** stored as the `GHCR_TOKEN` repo secret (workaround for the `GITHUB_TOKEN` package permission issue).

### Published image

```bash
docker pull ghcr.io/adepuharshavardhan2001/vehicle-intent:latest
```

---

## Kubernetes Deployment

The `k8s/` folder contains production-ready manifests.

### Create the secret first

Do **not** commit `k8s/secret.yaml`. Create it locally:

```bash
kubectl create secret generic vehicle-secrets \
  --from-literal=mysql-root-password=<your-password> \
  --from-literal=database-url="mysql+pymysql://root:<your-password>@vehicle-mysql:3306/vehicle_logs"
```

### Apply the rest

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

The Deployment mounts the model from the container image and reads config from the ConfigMap + Secret.

---

## Design Decisions

- **DistilBERT over BERT-base** — 40% smaller, 2x faster inference, ~1% accuracy loss. Critical for edge/voice latency requirements.
- **Multi-stage Docker build** — build tools (`gcc`, `python3-dev`) stay out of the final image. Runtime image only ships what's needed.
- **CPU-only torch** — the production container doesn't need CUDA. Saves ~600 MB per image.
- **Model distributed via GitHub Releases** — avoids committing 236 MB to git and keeps the repo small.
- **Connection pooling** (`pool_pre_ping=True`, `pool_recycle=3600`) — prevents stale MySQL connections.
- **Non-root user in Docker** — security best practice.
- **Full K8s manifests** — Deployment, Service, ConfigMap, Secret, PVC. The Secret is a placeholder; real secrets are created via `kubectl`.

---

## Known Limitations

- **Text-only input** — audio preprocessing (speech-to-text) is out of scope.
- **No live retraining** — the model is static once deployed.
- **No rate limiting** on the API.
- **Single-worker deployment** — `uvicorn` runs with 1 worker; production would use `gunicorn` with multiple Uvicorn workers.

---

## What This Project Demonstrates

- **Applied ML** — fine-tuning transformer models on real classification problems
- **MLOps** — taking a model from training to production with Docker, CI/CD, and K8s
- **Model distribution** — handling large binary artifacts via GitHub Releases
- **Production engineering** — multi-stage Docker, connection pooling, non-root user, health checks, structured error handling
- **API design** — FastAPI with typed Pydantic contracts and auto-generated docs
- **Testing discipline** — unit + integration tests with mocked external services
- **CI/CD rigor** — real test gates, no bypasses, secure secrets

---

## License

MIT
