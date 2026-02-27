import logging
import os

import bentoml
import mlflow
import pandas as pd
from pydantic import BaseModel

from sqlalchemy import create_engine
from feast import FeatureStore, FeatureService

# -------------------- Constants --------------------
ARTIFACT_MODEL_NAME = "model"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# -------------------- Logging --------------------
logger = logging.getLogger("spaceflight_service")
logger.setLevel(logging.INFO)

# -------------------- Pydantic Models --------------------
class SpaceflightInput(BaseModel):
    engines: float
    passenger_capacity: int
    crew: float
    d_check_complete: bool
    moon_clearance_complete: bool
    iata_approved: bool
    company_rating: float
    review_scores_rating: float

class BatchScoringRequest(BaseModel):
    request_start_date: str
    request_end_date: str
class ModelURI(BaseModel):
    model_name: str
    model_version: int

# -------------------- BentoML Service --------------------
@bentoml.service(name="spaceflight_service")
class SpaceflightService:
    def __init__(self):
        self.bento_model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # -------------------- Model Import Endpoint --------------------
    @bentoml.api(route="/import_model")
    async def import_model(self, model_uri: ModelURI) -> dict:
        """
        Imports and loads a model from MLflow into BentoML.
        """
        try:
            await self._load_model(model_uri)
            msg = f"Model {model_uri.model_name}/{model_uri.model_version} is imported and loaded."
            logger.info(msg)
            return {"message": msg}
        except Exception as e:
            logger.exception(f"Error importing model: {e}")
            return {"error": str(e)}

    async def _load_model(self, model_uri: ModelURI) -> None:
        """
        Helper function to import and load MLflow model into BentoML.
        """
        imported_model = bentoml.mlflow.import_model(
            name=model_uri.model_name,
            model_uri=f"models:/{model_uri.model_name}/{model_uri.model_version}"
        )
        self.bento_model = bentoml.mlflow.load_model(imported_model)

    # -------------------- Prediction Endpoint --------------------
    @bentoml.api(route="/predict")
    def predict(self, input_data: SpaceflightInput) -> dict:
        """
        Makes a prediction using the loaded BentoML model.
        """
        if self.bento_model is None:
            msg = "No model loaded. Please import a model first."
            logger.warning(msg)
            return {"error": msg}

        try:
            df = pd.DataFrame.from_records([input_data.model_dump()])
            prediction = self.bento_model.predict(df)
            return {"prediction": prediction.tolist()}
        except Exception as e:
            logger.exception(f"Prediction error: {e}")
            return {"error": "Prediction failed", "details": str(e)}

    @bentoml.api(route="/batch-scoring")
    def batch_scoring(self, request: BatchScoringRequest) -> dict:
        """
        Get historical features and do batch-scoring
        """
        try:

            db_host = os.getenv('POSTGRES_HOST', 'postgres')
            db_port = os.getenv('POSTGRES_PORT', '5432')
            db_user = os.getenv('POSTGRES_USER', 'user')
            db_password = os.getenv('POSTGRES_PASSWORD', 'password')
            db_name = os.getenv('POSTGRES_DB', 'spaceflight_db')

            BATCH_SCORING_FEATURE_SERVICE = os.getenv("BATCH_SCORING_FEATURE_SERVICE", "spaceflight_feature_service_v1")

            store = FeatureStore(repo_path="")
            batch_scoring_feature_service = store.get_feature_service(BATCH_SCORING_FEATURE_SERVICE)

            # Retrieve historical data and perform batch scoring
            batch_scoring_start_date = request.request_start_date
            batch_scoring_end_date = request.request_end_date
            df = store.get_historical_features(
                features=batch_scoring_feature_service,
                start_date=pd.to_datetime(batch_scoring_start_date),
                end_date=pd.to_datetime(batch_scoring_end_date)
            ).to_df()

            if df.empty:
                return {"status": "success", "message": "No data found for the given date"}

            predictions = self.bento_model.predict(df)
            prediction_df = pd.DataFrame(columns=['shuttle_id', 'company_id', 'prediction', 'pred_timestamp'])

            # Add predictions
            prediction_df['shuttle_id'] = df['shuttle_id']
            prediction_df['company_id'] = df['company_id']
            prediction_df['prediction'] = predictions.tolist()
            prediction_df['pred_timestamp'] = df['event_timestamp']  # Timestamp di quando è avvenuto lo scoring)

            connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
            engine = create_engine(connection_string)
            prediction_df.to_sql('spaceflight_prediction_table', engine, if_exists='replace', index=False)

            return {
                "status": "success",
                "rows_processed": len(df),
                "table": "spaceflight_prediction_table"
            }

        except Exception as e:
            logger.exception(f"Batch scoring error: {e}")
            return {"error": "Batch scoring failed", "details": str(e)}