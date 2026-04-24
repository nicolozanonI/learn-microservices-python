FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY ./feature_repo /app/feature_repo

WORKDIR /app/feature_repo

EXPOSE 8888

ENV FEAST_USAGE=False

CMD ["feast", "ui", "-p", "8888", "-h", "0.0.0.0"]