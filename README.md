#  In-Vehicle Intent Classifier

An end-to-end Natural Language Processing (NLP) microservice that classifies driver voice commands into 183 distinct automotive intents (Navigation, Climate, Media, Communication, and Vehicle Controls). 
The system is fully containerized using Docker and logs every prediction to a MySQL database, mimicking a real-world production environment.

---

##  Performance Metrics

- **Model:** Fine-tuned DistilBERT (Hugging Face)
- **Dataset:** 1,500 labeled in-car utterances
- **Classes:** 183 unique intents
- **Validation Accuracy:** 79%
- **Deployment:** Fully containerized via Docker Compose

---

## 🛠️ Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.9** | Core programming language |
| **Transformers / PyTorch** | NLP model fine-tuning |
| **DistilBERT (base-uncased)** | Lightweight, efficient text classification model |
| **FastAPI / Uvicorn** | High-performance REST API framework |
| **MySQL 8.0** | Relational database for logging predictions |
| **SQLAlchemy / PyMySQL** | ORM and database connection |
| **Docker / Docker Compose** | Containerization and orchestration |
| **Pandas / NumPy** | Data manipulation and preprocessing |

---

##  Key Features

- **AI-Powered Intent Recognition:** Accurately classifies 183 distinct voice commands (e.g., `navigate_to_poi`, `set_temperature`, `play_music`).
- **FastAPI Backend:** Automatic interactive Swagger documentation (`/docs`).
- **Dockerized Deployment:** Runs seamlessly on any machine with Docker installed, eliminating environment conflicts.
- **Production Logging:** Automatically stores every prediction with a timestamp in a MySQL database for auditing and analysis.
- **Modular Architecture:** Clean separation of data, model, database, and API layers.

---

##  Architecture Overview

1. **User** sends a voice command (text) to the API.
2. **FastAPI** validates the request and passes it to the `model.py` wrapper.
3. **DistilBERT** model predicts the intent and confidence score.
4. **API** saves the input, predicted intent, confidence, and timestamp to **MySQL** via SQLAlchemy.
5. **API** returns the JSON response to the user.

---

##  Project Structure

```text
In-vehicle-classifier/
├── app/                  # FastAPI application
│   ├── main.py           # API endpoint definitions
│   ├── model.py          # Model loading and inference wrapper
│   ├── database.py       # SQLAlchemy database connection
│   ├── crud.py           # Database CRUD operations
│   └── schemas.py        # Pydantic request/response models
├── data/                 # training dataset (Excel)
├── models/               # Trained DistilBERT model and label map
├── train.py              # Python script to fine-tune the model
├── requirements.txt      # Python dependencies
├── Dockerfile            # Instructions to build the API container
├── docker-compose.yml    # Orchestrates API and MySQL containers
└── README.md             # Project documentation (This file)

How to Run Locally
Make sure you have Docker Desktop installed and running.

1. Clone the repository:

git clone https://github.com/Adepuharsvardhan2001/in-vehicle-intent-classifier.git
cd in-vehicle-intent-classifier

2. Train the model (Optional - Pre-trained model is included):
python train.py

3. Build and start the containers:

docker-compose up --build
4. Access the API:
Open your browser and go to http://localhost:8000/docs

 API Endpoints
1. Predict Intent
Endpoint: POST /predict

Purpose: Submit a voice command and receive the classified intent and confidence score.

Example Request Body:

json
{
  "text": "Navigate to the nearest gas station"
}

Example Response:

json
{
  "intent": "navigate_to_poi",
  "confidence": 0.98
}

2. Retrieve Logs

Endpoint: GET /logs

Purpose: Fetch the 10 most recent predictions stored in the MySQL database.

Example Response:

json
[
  {
    "id": 1,
    "command_text": "Navigate to the nearest gas station",
    "predicted_intent": "navigate_to_poi",
    "confidence": 0.98,
    "timestamp": "2026-06-24T13:53:02"
  }
]
