from fastapi import FastAPI, HTTPException
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import os
import logging
from pydantic import BaseModel

from utils.evidently_integration import project_setup, data_drift_check, model_performance_check

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KEDRO_API_URL = os.getenv("KEDRO_API_URL", "http://host.docker.internal:8005/run-pipeline")

np.random.seed(42)

app = FastAPI()

class CurrentRequest(BaseModel):
    drift: bool = False

def evidently_warnings(data_failed):
    logger.warning(f"{len(data_failed)} failed tests detected:")
    for t in data_failed:
        logger.warning(f"ID: {t.id}, \n"
                       f"Name: {t.name}, \n"
                       f"Column: {t.metric_config.params.get('column')},\n"
                       f"Metric type: {t.metric_config.params.get('type')},\n"
                       f"Threshold (drift/fixed test): {t.metric_config.params.get('drift_share') or t.test_config.get('threshold')}\n"
                       f"Critical test? {t.test_config.get('is_critical')}\n\n")

@app.get("/analyze")
def analyze():
    logger.info("Starting Evidently analysis...")
    project = project_setup()

    # --- Data drift and model performance checks ---
    logger.info("Checking data drift...")
    failed_data_tests = data_drift_check(project)
    if failed_data_tests:
        evidently_warnings(failed_data_tests)

    logger.info("Checking model performance...")
    failed_model_tests = model_performance_check(project)
    if failed_model_tests:
        evidently_warnings(failed_model_tests)

    # --- Trigger Kedro pipeline if any test failed ---
    if failed_data_tests or failed_model_tests:
        logger.info("Drift detected — trigger Kedro pipeline")

    return {
        "status": "OK",
        "message": "Evidently analysis completed",
        "failed_data_tests": len(failed_data_tests),
        "failed_model_tests": len(failed_model_tests),
    }
