FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY feature_repo /app/feature_repo
WORKDIR /app/feature_repo

CMD ["sh", "-c", "feast teardown && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
