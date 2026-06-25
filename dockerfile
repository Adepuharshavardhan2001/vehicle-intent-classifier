FROM python:3.9-slim

# Install system dependencies required for MySQL and Cryptography
RUN apt-get update && apt-get install -y gcc libmariadb-dev-compat libmariadb-dev build-essential python3-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY train.py .

# INSTALL EVERYTHING (WITH CRYPTOGRAPHY AND EXTRA TIMEOUT)
RUN pip install --no-cache-dir --default-timeout=100 \
    fastapi==0.103.1 \
    uvicorn==0.23.2 \
    sqlalchemy==2.0.20 \
    pymysql==1.0.2 \
    pandas==2.0.3 \
    openpyxl==3.1.2 \
    numpy==1.24.3 \
    transformers==4.35.0 \
    torch==2.1.2 \
    scikit-learn==1.3.0 \
    python-dotenv==1.0.0 \
    pydantic==2.3.0 \
    tqdm==4.66.1 \
    cryptography==42.0.5 \
    --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]