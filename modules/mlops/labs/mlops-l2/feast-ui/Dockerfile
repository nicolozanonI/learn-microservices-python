FROM python:3.13-slim

WORKDIR /app

# Installa feast e i driver per Postgres (necessari per leggere il registry)
RUN pip install --no-cache-dir feast[s3] boto3

# Copiamo solo la cartella del repository delle feature
COPY ./feature_repo /app/feature_repo

WORKDIR /app/feature_repo

# Espone la porta di default della UI
EXPOSE 8888

# Avvia la UI specificando l'host per renderla accessibile dall'esterno
CMD ["feast", "ui", "-p", "8888", "-h", "0.0.0.0"]