
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import mlflow
from mlflow.models import infer_signature

from feast import FeatureStore, FeatureService
from feast.infra.offline_stores.offline_store import RetrievalJob
from datetime import datetime, timezone
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import SavedDatasetPostgreSQLStorage


def get_latest_dataset_metadata(store):
    try:
        dataset_list = store.list_saved_datasets()
    except Exception:
        dataset_list = []

    default_start = "2026-03-16 00:00:00+00:00"  # ora è aware
    default_end = datetime.now(tz=timezone.utc).isoformat()  # es. "2026-03-19T15:41:45.123456+00:00"

    valid_datasets = [
        ds for ds in dataset_list
        if ds.tags.get('start_date') and ds.tags.get('end_date')
    ]

    if not valid_datasets:
        return False, default_start, default_end

    try:
        latest_ds = max(
            valid_datasets,
            key=lambda x: pd.to_datetime(x.tags.get('end_date'), utc=True)
        )
        return True, latest_ds.tags.get('start_date'), latest_ds.tags.get('end_date')

    except Exception:
        return False, default_start, default_end


def split_data(start_date: str, end_date: str, parameters: dict) -> tuple:
    """
    Usa Feast come prima, ma prende start_date/end_date dall'esterno (Kedro params),
    poi genera training_df con get_historical_features e fa train/test split.
    """
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    TRAINING_FEATURE_SERVICE = os.getenv("TRAINING_FEATURE_SERVICE", "spaceflight_feature_service_v1")

    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment("spaceflights-kedro")
    mlflow.start_run(run_name="training")
    mlflow.set_tag("Training Info", "RandomForest model for spaceship data")

    mlflow.log_params(parameters)
    mlflow.log_param("start_date", start_date)
    mlflow.log_param("end_date", end_date)

    feature_store_path = os.path.join(os.getcwd(), ".")
    store = FeatureStore(repo_path=feature_store_path)

    spaceflight_features = store.get_feature_service(TRAINING_FEATURE_SERVICE)

    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)

    # Retrieval storico per training (Feast pattern)
    # Feast supporta training data generation via get_historical_features(...) [1](https://docs.feast.dev/getting-started/concepts/feature-retrieval)
    training_job = store.get_historical_features(
        features=spaceflight_features,
        start_date=start_dt,
        end_date=end_dt,
    )

    training_df = training_job.to_df()

    # (opzionale) log dataset a MLflow come artifact
    csv_path = "./data/05_model_input/training_input_table.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    training_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path="datasets")

    X = training_df[parameters["features"]]
    y = training_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
    )

    return X_train, X_test, y_train, y_test, training_df, training_job



def train_model(X_train: pd.DataFrame, y_train: pd.Series, training_df: pd.DataFrame, training_job: RetrievalJob, parameters: dict) -> RandomForestRegressor:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
        parameters: Parameters defined in parameters/data_science.yml.

    Returns:
        Trained model.
    """

    regressor = RandomForestRegressor(**parameters['regressor'])
    regressor.fit(X_train, y_train)

    signature = infer_signature(X_train, regressor.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=regressor,
        name="model",
        signature=signature,
        input_example=X_train,
        registered_model_name="spaceflights-kedro",
    )

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(
        filter_string="name='spaceflights-kedro'",
        max_results=1000,  # o un numero ragionevole
    )

    if not versions:
        raise ValueError("Nessuna model version trovata per 'spaceflights-kedro' in MLflow Registry.")

    # mv.version è stringa -> cast a int per confronto robusto
    mv = max(versions, key=lambda v: int(v.version))
    model_name = mv.name
    model_version = mv.version
    store = FeatureStore(repo_path=os.getcwd())
    run_id = mlflow.active_run().info.run_id

    saved_dataset = store.create_saved_dataset(
        from_=training_job,  # ✅ QUI
        name=f"training_dataset_{model_version}",
        storage=SavedDatasetPostgreSQLStorage(
            table_ref="training_dataset_table"
        ),
        tags={
            "model_name": model_name,
            "model_version": str(model_version),
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "start_date": parameters.get("start_date", ""),
            "end_date": parameters.get("end_date", ""),
            "mlflow_run_id": run_id,
        },
    )

    return regressor


def evaluate_model(
    regressor: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Calculates and logs model details.

    Args:
        regressor: Trained model.
        X_train: Training data of independent features.
        y_train: Training data for price.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
        parameters: Parameters defined in parameters/data_science.yml.
    """

    # Log metrics
    y_pred = regressor.predict(X_test)
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("rmse", mean_squared_error(y_test, y_pred))
    mlflow.end_run()

"""
def export_model_to_bentoml(regressor):
    Exports the trained model to BentoML.

    Args:
        regressor: Trained model.

    Returns:
        Info about the saved model in BentoML.
    model_name = "spaceflights-pandas"
    model_info = bentoml.sklearn.save_model(
        name=model_name,
        model=regressor,
        signatures={"predict": {"batchable": True}},
    )
    return model_info"""
