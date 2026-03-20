FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir 'feast[s3, postgres]' boto3 psycopg2-binary sqlalchemy-utils

COPY ./feature_repo /app/feature_repo

WORKDIR /app/feature_repo

EXPOSE 8888

ENV FEAST_USAGE=False

CMD ["feast", "ui", "-p", "8888", "-h", "0.0.0.0"]