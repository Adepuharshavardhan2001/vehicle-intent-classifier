#  In-Vehicle Intent Classifier

A production-ready ML microservice that classifies in-vehicle voice commands using a fine-tuned DistilBERT model. Built with FastAPI, MySQL, Docker, and deployed via CI/CD to Kubernetes.

![CI](https://github.com/Adepuharshavardhan2001/vehicle-intent-classifier/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/Adepuharshavardhan2001/vehicle-intent-classifier/actions/workflows/cd.yml/badge.svg)

---

## 📋 Features

- **Intent Classification** — Fine-tuned DistilBERT model for 5+ in-vehicle intents
- **REST API** — FastAPI with automatic Swagger documentation
- **Prediction Logging** — MySQL database stores all predictions with timestamps
- **Health Checks** — Kubernetes-ready liveness and readiness endpoints
- **Containerized** — Docker multi-stage build for production
- **CI/CD Pipeline** — GitHub Actions for automated testing, building, and deployment
- **Kubernetes Ready** — Deployment, Service, ConfigMap, and Secret manifests included

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Framework** | PyTorch, HuggingFace Transformers |
| **Model** | DistilBERT (fine-tuned) |
| **API** | FastAPI, Uvicorn |
| **Database** | MySQL 8.0, SQLAlchemy |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Orchestration** | Kubernetes |
| **Registry** | GitHub Container Registry (GHCR) |

---

##  Project Structure

vehicle-intent-classifier/

├── .github/

│ └── workflows/

│ ├── ci.yml # CI Pipeline (lint, test, build)

│ └── cd.yml # CD Pipeline (build, push, deploy)

├── app/

│ ├── main.py # FastAPI application

│ ├── model.py # ML model wrapper

│ ├── database.py # Database connection & model

│ ├── crud.py # Database operations

│ ├── schema.py # Pydantic request/response schemas

│ └── tests/

│ ├── test_main.py # API tests

│ └── test_model.py # Model tests

├── k8s/

│ ├── deployment.yaml # Kubernetes Deployments

│ ├── service.yaml # Kubernetes Services

│ ├── configmap.yaml # Environment configuration

│ ├── secret.yaml # Sensitive data

│ └── pvc.yaml # Persistent storage

├── models/

│ ├── distilbert_finetuned/ # Trained model files

│ └── label_map.json # Intent label mapping

├── train.py # Model training script

├── Dockerfile # Container build instructions

├── docker-compose.yml # Local orchestration

├── requirements.txt # Production dependencies

├── requirements-dev.txt # Development dependencies

└── README.md # This file
