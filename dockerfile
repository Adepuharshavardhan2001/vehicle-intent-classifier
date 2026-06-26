FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential python3-dev libmariadb-dev-compat libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

RUN useradd --create-home appuser
RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]