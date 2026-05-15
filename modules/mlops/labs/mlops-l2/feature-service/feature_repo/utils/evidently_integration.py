import datetime
from datetime import timedelta, timezone
import os
import logging
from typing import Optional, Tuple, List

import pandas as pd
from feast import FeatureStore

from evidently.ui.workspace import RemoteWorkspace
from evidently import Report, DataDefinition, Dataset, Regression
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset
from evidently.metrics import MAE
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot

logger = logging.getLogger(__name__)

# -------------------- Env (MinIO) --------------------
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
os.environ["AWS_ENDPOINT_URL"] = os.getenv("MINIO_URL", "http://minio:9000")

# -------------------- Constants --------------------
DATA_DRIFT_FS = os.getenv("DATA_DRIFT_FS", "spaceflight_feature_service_v1")
TRAINING_FV = os.getenv("TRAINING_FV", "spaceflight_features_view_v1")
MODEL_DRIFT_FS = os.getenv("MODEL_DRIFT_FS", "spaceflight_evidently_feature_view")

EVIDENTLY_WS_URL = os.getenv("EVIDENTLY_WS_URL", "http://evidently-ai:8000")
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", ".")

# -------------------- Clients --------------------
ws = RemoteWorkspace(EVIDENTLY_WS_URL)
store = FeatureStore(repo_path=FEAST_REPO_PATH)

# -------------------- Helpers --------------------
def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    """Ensure a datetime is timezone-aware UTC. If naive, assume UTC."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

def _get_model_eval_df(start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """
    Recupera target + prediction via Feast e li unisce nel range richiesto.
    Restituisce un DF che contiene almeno:
    shuttle_id, company_id, event_timestamp, price (target), prediction
    """
    target_df = _get_historical(DATA_DRIFT_FS, start, end)
    pred_df = _get_historical(MODEL_DRIFT_FS, start, end)

    logger.warning(f"[MODEL_EVAL] range={start}->{end} target_rows={len(target_df)} pred_rows={len(pred_df)}")
    logger.warning(f"[MODEL_EVAL] target_cols={list(target_df.columns)}")
    logger.warning(f"[MODEL_EVAL] pred_cols={list(pred_df.columns)}")

    df = pd.merge(pred_df, target_df, on=["shuttle_id", "company_id", "event_timestamp"], how="inner")
    logger.warning(f"[MODEL_EVAL] merged_rows={len(df)}")
    df = _localize_event_timestamp(df)
    return df
def _localize_event_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure event_timestamp column is UTC-aware."""
    if "event_timestamp" in df.columns:
        # se è datetime64[ns] naive -> localizza in UTC
        if hasattr(df["event_timestamp"].dt, "tz") and df["event_timestamp"].dt.tz is None:
            df["event_timestamp"] = df["event_timestamp"].dt.tz_localize("UTC")
    return df

def _get_historical(feature_service_name: str, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """
    Fetch historical features from Feast for a datetime range using a FeatureService.
    """
    df = store.get_historical_features(
        features=store.get_feature_service(feature_service_name),
        start_date=_ensure_utc(start),
        end_date=_ensure_utc(end),
    ).to_df()
    return _localize_event_timestamp(df)

def _add_drift_dashboard_panels(project) -> None:
    panels = [
        DashboardPanelPlot(
            title="Price and passenger capacity drift",
            subtitle="Drift numerico su prezzo e capacità passeggeri",
            size="half",
            values=[
                PanelMetric(legend="Price drift", metric="ValueDrift", metric_labels={"column": "price"}),
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
                PanelMetric(legend="Moon Clearance", metric="ValueDrift", metric_labels={"column": "moon_clearance_complete"}),
                PanelMetric(legend="IATA Approved", metric="ValueDrift", metric_labels={"column": "iata_approved"}),
            ],
            plot_params={"plot_type": "bar"},
        ),
    ]
    for panel in panels:
        project.dashboard.add_panel(panel, tab="Data Drift")

def check_failed_tests(my_eval) -> list:
    return [t for t in my_eval.tests_results if t.status == "FAIL"]

# -------------------- Public API: Streamlit-oriented --------------------
def project_setup():
    """
    Crea un nuovo project Evidently (come prima).
    """
    project = ws.create_project(
        f"My Project {datetime.datetime.now(tz=timezone.utc).strftime('%Y %m %d - %H %M %S')}"
    )
    project.description = "My production monitoring simulation project."
    project.save()
    return project

def get_datasets_from_ranges(
    feature_service: str,
    schema: DataDefinition,
    ref_start: datetime.datetime,
    ref_end: datetime.datetime,
    cur_start: datetime.datetime,
    cur_end: datetime.datetime,
) -> Tuple[Dataset, Dataset]:
    """
    Reference e Current vengono scelti dalla UI (Streamlit).
    Nessun saved_dataset: usiamo Feast get_historical_features su range.
    """
    ref_df = _get_historical(feature_service, _ensure_utc(ref_start), _ensure_utc(ref_end))
    cur_df = _get_historical(feature_service, _ensure_utc(cur_start), _ensure_utc(cur_end))

    logger.info(
        f"Reference: {ref_df['event_timestamp'].min() if 'event_timestamp' in ref_df else None} "
        f"→ {ref_df['event_timestamp'].max() if 'event_timestamp' in ref_df else None} "
        f"({len(ref_df)} record)"
    )
    logger.info(
        f"Current:   {cur_df['event_timestamp'].min() if 'event_timestamp' in cur_df else None} "
        f"→ {cur_df['event_timestamp'].max() if 'event_timestamp' in cur_df else None} "
        f"({len(cur_df)} record)"
    )

    return (
        Dataset.from_pandas(ref_df, data_definition=schema),
        Dataset.from_pandas(cur_df, data_definition=schema),
    )

def data_drift_schema_from_feature_view() -> DataDefinition:
    """
    Ricava DataDefinition dal FeatureView TRAINING_FV (come prima).
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

    return DataDefinition(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )

def report_data_drift(
    schema: DataDefinition,
    project,
    ref_start: datetime.datetime,
    ref_end: datetime.datetime,
    cur_start: datetime.datetime,
    cur_end: datetime.datetime,
):
    """
    Identico nei calcoli Evidently (stessi preset/report/panels),
    ma reference/current arrivano dalla UI.
    """
    eval_data_ref, eval_data_prod = get_datasets_from_ranges(
        DATA_DRIFT_FS, schema, ref_start, ref_end, cur_start, cur_end
    )

    # --- stessi report di prima ---
    for report_obj, tags in [
        (Report([DataDriftPreset(drift_share=0.7)], include_tests=True), ["Data drift present", "tests included"]),
        (Report([DataSummaryPreset()]), ["Data summary present"]),
    ]:
        ws.add_run(
            project.id,
            report_obj.run(reference_data=eval_data_ref, current_data=eval_data_prod, tags=tags),
            include_data=False,
        )

    eval_drift = Report([DataDriftPreset(drift_share=0.7)], include_tests=True).run(
        reference_data=eval_data_ref,
        current_data=eval_data_prod,
        tags=["Data drift present", "tests included"],
    )

    _add_drift_dashboard_panels(project)

    # Prima usavi questa parte per decidere se salvare un saved_dataset.
    # La lasciamo "logicamente" intatta ma NON salviamo nulla.
    ref_df = eval_data_ref.as_dataframe()
    curr_df = eval_data_prod.as_dataframe()

    has_drift = bool(check_failed_tests(eval_drift))
    start_date = curr_df["event_timestamp"].min() if (has_drift and "event_timestamp" in curr_df) else (
        ref_df["event_timestamp"].min() if "event_timestamp" in ref_df else None
    )
    end_date = curr_df["event_timestamp"].max() if ("event_timestamp" in curr_df) else None

    logger.info(f"has_drift={has_drift}, drift_window_start={start_date}, drift_window_end={end_date}")

    return eval_drift

def report_model_drift(
    schema: DataDefinition,
    project,
    ref_start: datetime.datetime,
    ref_end: datetime.datetime,
    cur_start: datetime.datetime,
    cur_end: datetime.datetime,
):
    """
    Calcoli Evidently invariati (RegressionPreset + MAE),
    ma reference/current sono costruiti direttamente dai range scelti in UI.
    Niente split.
    """
    ref_start = _ensure_utc(ref_start)
    ref_end   = _ensure_utc(ref_end)
    cur_start = _ensure_utc(cur_start)
    cur_end   = _ensure_utc(cur_end)

    ref_df = _get_model_eval_df(ref_start, ref_end)
    cur_df = _get_model_eval_df(cur_start, cur_end)

    # Guard rail: se uno dei due è vuoto, RegressionPreset/MAE non può funzionare
    if ref_df.empty or cur_df.empty:
        missing = []
        if ref_df.empty:
            missing.append("reference")
        if cur_df.empty:
            missing.append("current")
        raise ValueError(
            f"Model performance: nessun record joinato per {', '.join(missing)} "
            f"(serve target+prediction nel range selezionato)."
        )

    eval_data_ref  = Dataset.from_pandas(ref_df, data_definition=schema)
    eval_data_prod = Dataset.from_pandas(cur_df, data_definition=schema)

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

def data_drift_check(
    project,
    ref_start: datetime.datetime,
    ref_end: datetime.datetime,
    cur_start: datetime.datetime,
    cur_end: datetime.datetime,
) -> list:
    """
    Retrieve features from Feast and check data drift with Evidently.
    (calcoli invariati; cambia solo come scegli reference/current)
    """
    schema = data_drift_schema_from_feature_view()
    snapshot = report_data_drift(schema, project, ref_start, ref_end, cur_start, cur_end)
    return check_failed_tests(snapshot)

def model_performance_check(
    project,
    ref_start: datetime.datetime,
    ref_end: datetime.datetime,
    cur_start: datetime.datetime,
    cur_end: datetime.datetime,
) -> list:
    schema = DataDefinition(
        regression=[Regression(name="default", target="price", prediction="prediction")]
    )
    snapshot = report_model_drift(schema, project, ref_start, ref_end, cur_start, cur_end)
    return check_failed_tests(snapshot)