import datetime
from datetime import timedelta, timezone
import os
import pandas as pd
from evidently.ui.workspace import Workspace, Snapshot, RemoteWorkspace
from evidently import Report, DataDefinition, Dataset, Regression
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset
from evidently.metrics import MAE
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot
from feast import FeatureStore
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import SavedDatasetPostgreSQLStorage
import logging

logger = logging.getLogger(__name__)

os.environ["AWS_ACCESS_KEY_ID"]     = os.getenv("MINIO_ROOT_USER", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
os.environ["AWS_ENDPOINT_URL"]      = os.getenv("MINIO_URL", "http://minio:9000")

DATA_DRIFT_FS  = os.getenv("DATA_DRIFT_FS",  "spaceflight_feature_service_v1")
TRAINING_FV    = os.getenv("TRAINING_FV",    "spaceflight_features_view_v1")
MODEL_DRIFT_FS = os.getenv("MODEL_DRIFT_FS", "spaceflight_evidently_feature_view")
HISTORY_START  = datetime.datetime(2025, 6, 6, tzinfo=timezone.utc)

#ws = Workspace.create(path="s3://evidently-ai/workspace")
ws    = RemoteWorkspace("http://evidently-ai:8000")
store = FeatureStore(repo_path=".")

# Stato condiviso tra le funzioni
_state: dict = {
    "split_timestamp": datetime.datetime(2025, 1, 1, tzinfo=timezone.utc),
}


def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    """Ensure a datetime is timezone-aware UTC. If naive, assume UTC."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _localize_event_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure event_timestamp column is UTC-aware."""
    if df['event_timestamp'].dt.tz is None:
        df['event_timestamp'] = df['event_timestamp'].dt.tz_localize('UTC')
    return df


def _get_ts_from_tag(dataset, tag_key) -> pd.Timestamp | None:
    try:
        ts_str = dataset.tags.get(tag_key)
        logger.info(f"_get_ts_from_tag [{tag_key}] raw string: repr={repr(ts_str)}")
        result = pd.to_datetime(ts_str, utc=True) if ts_str else None
        logger.info(f"_get_ts_from_tag [{tag_key}] parsed: {result}")
        return result
    except (ValueError, TypeError):
        return None


def _get_historical(feature_service_name: str, start: datetime.datetime,
                    end: datetime.datetime) -> pd.DataFrame:
    df = store.get_historical_features(
        features=store.get_feature_service(feature_service_name),
        start_date=_ensure_utc(start),
        end_date=_ensure_utc(end),
    ).to_df()
    return _localize_event_timestamp(df)


def _save_dataset(start_date: datetime.datetime, end_date: datetime.datetime,
                  feature_service_name: str) -> None:
    """
    Save the new training dataset metadata on Feast.
    """
    start_date = _ensure_utc(start_date)
    end_date   = _ensure_utc(end_date)

    table_ref    = "training_" + end_date.strftime('%Y%m%d_%H%M%S')
    training_job = store.get_historical_features(
        features=store.get_feature_service(feature_service_name),
        start_date=start_date,
        end_date=end_date,
    )
    store.create_saved_dataset(
        from_=training_job,
        allow_overwrite=True,
        name="training_dataset_" + end_date.strftime('%Y%m%d_%H%M%S'),
        storage=SavedDatasetPostgreSQLStorage(table_ref=table_ref),
        tags={
            "type":       "training_dataset",
            "start_date": start_date.isoformat(),
            "end_date":   end_date.isoformat(),
        },
    )


def project_setup():
    project = ws.create_project(
        f"My Project {datetime.datetime.now(tz=timezone.utc).strftime('%Y %m %d - %H %M %S')}"
    )
    project.description = "My production monitoring simulation project."
    project.save()
    return project


def get_datasets(feature_service: str, schema: DataDefinition) -> tuple[Dataset, Dataset]:
    valid_datasets = [
        ds for ds in store.list_saved_datasets()
        if _get_ts_from_tag(ds, 'end_date') is not None
    ]
    if not valid_datasets:
        raise ValueError("Nessun saved dataset trovato con il tag 'end_date'.")

    latest    = max(valid_datasets, key=lambda x: _get_ts_from_tag(x, 'end_date'))
    ref_start = _get_ts_from_tag(latest, 'start_date')
    ref_end   = _get_ts_from_tag(latest, 'end_date')

    _state["split_timestamp"] = ref_end + timedelta(microseconds=1)

    df = _get_historical(
        feature_service,
        ref_start.to_pydatetime(),
        datetime.datetime.now(tz=timezone.utc)
    )

    logger.info(f"ref_end type: {type(ref_end)}, value: {ref_end}")
    logger.info(f"ref_end tzinfo: {ref_end.tzinfo}")
    logger.info(f"event_timestamp dtype dopo fetch: {df['event_timestamp'].dtype}")
    logger.info(f"event_timestamp sample: {df['event_timestamp'].iloc[0]}")
    logger.info(f"confronto diretto: {df['event_timestamp'].iloc[0] <= ref_end}")

    ref_df  = df[df['event_timestamp'] <= ref_end]
    curr_df = df[df['event_timestamp'] >= _state["split_timestamp"]]

    logger.info(
        f"Reference:  {ref_df['event_timestamp'].min()} → {ref_df['event_timestamp'].max()} ({len(ref_df)} record)")
    logger.info(
        f"Current:    {curr_df['event_timestamp'].min()} → {curr_df['event_timestamp'].max()} ({len(curr_df)} record)")
    logger.info(f"company_rating nulli nel reference: {ref_df['company_rating'].isna().sum()}/{len(ref_df)}")

    return (
        Dataset.from_pandas(ref_df,  data_definition=schema),
        Dataset.from_pandas(curr_df, data_definition=schema),
    )


def _add_drift_dashboard_panels(project) -> None:
    panels = [
        DashboardPanelPlot(
            title="Price and passenger capacity drift",
            subtitle="Drift numerico su prezzo e capacità passeggeri",
            size="half",
            values=[
                PanelMetric(legend="Price drift",    metric="ValueDrift", metric_labels={"column": "price"}),
                PanelMetric(legend="Capacity drift", metric="ValueDrift", metric_labels={"column": "passenger_capacity"}),
            ],
            plot_params={"plot_type": "bar"},
        ),
        DashboardPanelPlot(
            title="Drift on categorical features",
            subtitle="Comparazione su colonne categoriche binarie",
            size="half",
            values=[
                PanelMetric(legend="D-Check Complete", metric="ValueDrift", metric_labels={"column": "d_check_complete"}),
                PanelMetric(legend="Moon Clearance",   metric="ValueDrift", metric_labels={"column": "moon_clearance_complete"}),
                PanelMetric(legend="IATA Approved",    metric="ValueDrift", metric_labels={"column": "iata_approved"}),
            ],
            plot_params={"plot_type": "bar"},
        ),
    ]
    for panel in panels:
        project.dashboard.add_panel(panel, tab="Data Drift")


def report_data_drift(schema: DataDefinition, project) -> Snapshot:
    eval_data_ref, eval_data_prod = get_datasets(DATA_DRIFT_FS, schema)

    for report_obj, tags in [
        (Report([DataDriftPreset(drift_share=0.7)], include_tests=True), ["Data drift present", "tests included"]),
        (Report([DataSummaryPreset()]),                                   ["Data summary present"]),
    ]:
        ws.add_run(project.id,
                   report_obj.run(reference_data=eval_data_ref, current_data=eval_data_prod, tags=tags),
                   include_data=False)

    eval_drift = Report([DataDriftPreset(drift_share=0.7)], include_tests=True).run(
        reference_data=eval_data_ref, current_data=eval_data_prod,
        tags=["Data drift present", "tests included"]
    )

    _add_drift_dashboard_panels(project)

    ref_df  = eval_data_ref.as_dataframe()
    curr_df = eval_data_prod.as_dataframe()

    has_drift  = bool(check_failed_tests(eval_drift))
    start_date = curr_df['event_timestamp'].min() if has_drift else ref_df['event_timestamp'].min()
    end_date   = curr_df['event_timestamp'].max()

    _save_dataset(start_date.to_pydatetime(), end_date.to_pydatetime(), DATA_DRIFT_FS)

    return eval_drift


def report_model_drift(schema: DataDefinition, project) -> Snapshot:
    split = _state["split_timestamp"]
    start = datetime.datetime(2025, 12, 12, tzinfo=timezone.utc)

    target_df     = _get_historical(DATA_DRIFT_FS,  start, datetime.datetime.now(tz=timezone.utc))
    prediction_df = _get_historical(MODEL_DRIFT_FS, start, datetime.datetime.now(tz=timezone.utc))

    df = pd.merge(prediction_df, target_df, on=['shuttle_id', 'company_id', 'event_timestamp'], how='inner')
    df = _localize_event_timestamp(df)

    eval_data_ref  = Dataset.from_pandas(df[df['event_timestamp'] <  split], data_definition=schema)
    eval_data_prod = Dataset.from_pandas(df[df['event_timestamp'] >= split], data_definition=schema)

    for report_obj, tags in [
        (Report([RegressionPreset()], include_tests=False), ["Regression present"]),
        (Report([MAE()],              include_tests=True),  ["MAE", "tests included"]),
    ]:
        run = report_obj.run(reference_data=eval_data_ref, current_data=eval_data_prod, tags=tags)
        ws.add_run(project.id, run, include_data=False)

    mape = Report([MAE()], include_tests=True).run(
        reference_data=eval_data_ref, current_data=eval_data_prod, tags=["MAE", "tests included"]
    )
    return mape


def check_failed_tests(my_eval: Snapshot) -> list:
    return [t for t in my_eval.tests_results if t.status == "FAIL"]


def data_drift_check(project) -> list:
    """
    Retrieve categorical and numerical features from Feast and check data drift with evidentlyAI.
    """
    fv = store.get_feature_view(TRAINING_FV)
    numerical_columns = [
        f.name
        for f in fv.schema
        if f.tags.get("type") in ["training_feature", "target_feature"]
        and f.tags.get("definition") == "numerical"
    ]
    categorical_columns = [
        f.name
        for f in fv.schema
        if f.tags.get("type") in ["training_feature", "target_feature"]
        and f.tags.get("definition") == "categorical"
    ]
    schema = DataDefinition(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )
    return check_failed_tests(report_data_drift(schema, project))


def model_performance_check(project) -> list:
    schema = DataDefinition(
        regression=[Regression(name="default", target="price", prediction="prediction")]
    )
    return check_failed_tests(report_model_drift(schema, project))