# ============================================================
# Stage 1: Build — install Python packages
# ============================================================
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ============================================================
# Stage 2: Runtime — small, self-contained image
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmariadb3 wget unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY app/ ./app/

RUN mkdir -p models && \
    wget -q https://github.com/Adepuharshavardhan2001/vehicle-intent-classifier/releases/download/v1.0/distilbert_finetuned.zip -O /tmp/model.zip && \
    unzip -q /tmp/model.zip -d models/ && \
    rm /tmp/model.zip && \
    apt-get purge -y wget unzip && \
    apt-get autoremove -y

RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]