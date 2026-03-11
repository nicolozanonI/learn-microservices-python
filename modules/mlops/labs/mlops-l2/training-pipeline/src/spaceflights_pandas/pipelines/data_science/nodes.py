
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import mlflow
from mlflow.models import infer_signature

from feast import FeatureStore, FeatureService
from datetime import datetime
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import SavedDatasetPostgreSQLStorage


def get_latest_dataset_metadata(store):
    """
    Check on Feast which is the training dataset, if there's any.
    """
    try:
        dataset_list = store.list_saved_datasets()
    except Exception:
        dataset_list = []

    default_start = "2025-01-01 00:00:00"
    default_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    valid_datasets = [
        ds for ds in dataset_list
        if ds.tags.get('start_date') and ds.tags.get('end_date')
    ]

    if not valid_datasets:
        return False, default_start, default_end

    try:
        latest_ds = max(
            valid_datasets,
            key=lambda x: pd.to_datetime(x.tags.get('end_date'))
        )

        start_date = latest_ds.tags.get('start_date')
        end_date = latest_ds.tags.get('end_date')

        return True, start_date, end_date

    except Exception:
        return False, default_start, default_end

def split_data(datal: pd.DataFrame, parameters: dict) -> tuple:
    """Get features from Feast and splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    START_DATE = os.getenv("START_DATE", "2025-12-20 00:00:00")
    TRAINING_FEATURE_SERVICE = os.getenv("TRAINING_FEATURE_SERVICE", "spaceflight_feature_service_v1")

    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment("spaceflights-kedro")
    mlflow.start_run(run_name="__default__")
    mlflow.set_tag("Training Info", "RandomForest model for spaceship data")
    mlflow.log_params(parameters)

    feature_store_path = os.path.join(os.getcwd(), ".")
    store = FeatureStore(repo_path=feature_store_path)

    spaceflight_features = store.get_feature_service(TRAINING_FEATURE_SERVICE)

    check_dataset, start_date, end_date = get_latest_dataset_metadata(store)

    #end_date = datetime.now()
    training_job = store.get_historical_features(
        features=spaceflight_features,
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date)
    )

    if check_dataset == False:
        table_ref = "training_" + datetime.fromisoformat(end_date).strftime('%Y%m%d_%H%M%S')
        dataset_storage = SavedDatasetPostgreSQLStorage(table_ref=table_ref)
        store.create_saved_dataset(from_=training_job,
                                   name="training_dataset_" + datetime.fromisoformat(end_date).strftime('%Y%m%d_%H%M%S'),
                                   storage=dataset_storage,
                                   tags={
                                       "type": "training_dataset",
                                       "start_date": start_date,
                                       "end_date": end_date
                                   })

    training_df = training_job.to_df()

    csv_path = "./data/05_model_input/final_input_table.csv"
    training_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path="datasets")

    X = training_df[parameters["features"]]
    y = training_df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> RandomForestRegressor:
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
