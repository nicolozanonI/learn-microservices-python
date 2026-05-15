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

## Feature service

### Using UI

Open the feature service UI at:

```
http://localhost:8501/
```

Then click on "Feast apply" button to create the registry and to populate it with all the features defined in ```spaceflight_features.py```:

### Using CLI

Alternatively, you can open a shell in the **feature-service**:

```bash
docker compose exec feature-service bash
```
and launch the following commands:

```bash
feast apply
exit
```

You can verify the feature store infrastructure, with all the entities, feature views and feature services at:

```
http://localhost:8888/p/spaceflight_project
```

---

## Dataset generation and model training

Optionally, explore and experiment manually using the notebooks in:

```
training-pipeline/notebooks
```

Change the values range of the new samples with the sliders and choose a month, then click "Generate" to create a new dataset. Then you can click 
"Train/retrain" to train a new model on that dataset

Verify that a new model appears in the MLflow Model Registry:

```
http://localhost:5000/#/models
```

---

## Load a model into the inference engine

Ask the inference service to load a specific model version:

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

---

## Run a test prediction

Send a sample prediction request to the inference service:

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

---


### Batch scoring

Classification labels are produced by the **inference-bentoml** service with:

```bash
curl -X POST http://localhost:3000/batch-scoring \
         -H "Content-Type: application/json" \
         -d '{
               "start_date": "<batch_scoring_start_date>",
               "end_date": "<batch_scoring_end_date>"
             }'  
```

*batch_scoring_start_date* and *batch_scoring_end_date* are given in output by the **feature-service** after each 
samples generation, under the buttons.

---

### Analyze data drift and performance changes

From the feature service UI, you can run the analysis comparing reference and current datasets:
1) Select a new month and generate a new dataset.
2) Copy the API call and perform batch-scoring on the new dataset
3) Select the month that you consider reference, click the "Analyze" button and select a second month that will be used as current dataset
4) Finally, open **Evidently AI** and inspect the results in the UI. Reference dataset is the one used in the
**training-pipeline**, current dataset is the one generated with the **feature-service**

