FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir feast[s3] boto3

COPY ./feature_repo /app/feature_repo

WORKDIR /app/feature_repo

EXPOSE 8888

CMD ["feast", "ui", "-p", "8888", "-h", "0.0.0.0"]