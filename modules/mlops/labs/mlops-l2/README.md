# Setup

## Python environment

Create and activate a dedicated Python environment using **pyenv**:

```bash
pyenv install 3.13.3
pyenv local 3.13.3
pyenv virtualenv mlops
pyenv activate mlops

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Build the inference service (BentoML)

Move to the BentoML inference project and build the service:

```bash
cd inference-bentoml
bentoml build
```

This command outputs a service identifier, for example:

```
spaceflight_service:h3f3atgjikqxhigt
```

Use the generated identifier to build the container image:

```bash
bentoml containerize spaceflight_service:h3f3atgjikqxhigt
```

> ⚠️ Replace the example tag with the one generated on your machine.

---

## Update `docker-compose.yml`

Edit the `inference` service definition to reference the newly built image:

```yaml
inference:
  image: spaceflight_service:h3f3atgjikqxhigt   # ← your generated image name:tag
  pull_policy: never
  environment:
    - MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
  depends_on:
    - mlflow
  ports:
    - "3000:3000"
  healthcheck:
    test: [ "CMD", "curl", "-f", "http://localhost:3000/healthz" ]
    interval: 20s
    timeout: 15s
    retries: 10
```

Start the full stack:

```bash
docker compose up --build -d
```

---

# Usage

## Feature Service

### Using UI

Open the Feature Service UI:
```
http://localhost:8501/
```
Click on "Feast apply" to:
- create the Feature Store registry
- register all features defined in spaceflight_features.py


### Using CLI

Alternatively, run the command inside the container:
```
docker compose exec feature-service bash
```
Then:
```
feast apply
exit
```

### Verify Feature Store

You can inspect entities, feature views and feature services at:
```
http://localhost:8888/p/spaceflight_project
```

-----------------------------------------------------

## Dataset Generation & Model Training

### Generate Dataset (UI)

1. Open the UI:
```
   http://localhost:8501/
```
2. Configure:
   - number of samples
   - feature ranges (sliders)

3. Select one month

4. Click "Generate"

The service will generate a dataset forthe selected motnh and will store it on the Offline Store.
### Train / Retrain Model

After generating a dataset:

1. Select the generated month
2. Click "Train"

This will trigger the training pipeline and the trained model will be saved on the model registry.


### Verify Model in MLflow

Check that a new model is registered:
```
http://localhost:5000/#/models
```

### Optional: Notebooks

You can also explore manually:
```
training-pipeline/notebooks
```

-----------------------------------------------------

## Load Model into Inference Service

Load a specific model version:
```bash
curl -X POST http://localhost:3000/import_model \
  -H "Content-Type: application/json" \
  -d '{
        "model_uri": {
          "model_name": "spaceflights-kedro",
          "model_version": 1
        }
      }'
```

-----------------------------------------------------

## Run a Single Prediction
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "input_data": {
          "engines": 2.0,
          "passenger_capacity": 5,
          "crew": 1.0,
          "d_check_complete": true,
          "moon_clearance_complete": true,
          "iata_approved": false,
          "company_rating": 4.5,
          "review_scores_rating": 4.7
        }
      }'
```

-----------------------------------------------------

## Batch Scoring

Run predictions over a time window:
```bash
curl -X POST http://localhost:3000/batch-scoring \
  -H "Content-Type: application/json" \
  -d '{
        "start_date": "<start_date>",
        "end_date": "<end_date>"
      }'
```
Note:
start_date and end_date are provided by the Feature Service UI: select a month and press "API call", then copy the call on your shell.


-----------------------------------------------------

## Analyze Data Drift & Model Performance

When you have trained and loaded a model, you can generate a new dataset on another month and run the analysis directly from the UI.
You have to perform batch-scoring also on the new generated data.

General workflow:

1. Generate a dataset (month A)
2. Train a model for that month, then load it on BentoML
3. Generate another dataset (month B)
4. Run batch scoring on both dataset (click "API call" on each month and copy it on your shell)
4. In the UI:
   - select month A → click "Analyze"
   - select month B as current

Important:
- Reference and current must be different months


### View Results in Evidently

Open:
```
http://localhost:8888
```
You can inspect:
- data drift
- model performance
- failed tests

