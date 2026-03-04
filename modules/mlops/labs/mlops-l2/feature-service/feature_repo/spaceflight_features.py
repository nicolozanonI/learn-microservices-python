from datetime import timedelta
import pandas as pd
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource, ValueType, FeatureStore,
)
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String, Bool, Int32
from typing import Dict, Any



shuttle = Entity(name="shuttle_id",join_keys=["shuttle_id"], description="Shuttle ID",
                 value_type=ValueType.INT32)
company = Entity(name="company_id", join_keys=["company_id"], description="Company ID",
                 value_type=ValueType.INT32)


spaceflight_test_source = PostgreSQLSource(
    name="shuttle_stats_postgres",
    table="spaceflight_table",
    timestamp_field="event_timestamp",
)


shuttle_features_view = FeatureView(
    name="spaceflight_features_view_v1",
    entities=[shuttle, company],
    ttl=None,
    schema=[
        Field(name="engines", dtype=Float64, tags={"type": "training_feature",
                                                   "definition": "numerical"}),
        Field(name="passenger_capacity", dtype=Int64, tags={"type": "training_feature",
                                                            "definition": "numerical"}),
        Field(name="crew", dtype=Float64, tags={"type": "training_feature",
                                                "definition": "numerical"}),
        Field(name="d_check_complete", dtype=Bool, tags={"type": "training_feature",
                                                         "definition": "categorical"}),
        Field(name="moon_clearance_complete", dtype=Bool, tags={"type": "training_feature",
                                                                "definition": "categorical"}),
        Field(name="iata_approved", dtype=Bool, tags={"type": "training_feature",
                                                      "definition": "categorical"}),
        Field(name="company_rating", dtype=Float64, tags={"type": "training_feature",
                                                          "definition": "numerical"}),
        Field(name="review_scores_rating", dtype=Float64, tags={"type": "training_feature",
                                                                "definition": "numerical"}),
        Field(name="price", dtype=Float64, tags={"type": "target_feature",
                                                 "definition": "numerical"}),
    ],
    online=True,
    source=spaceflight_test_source,
)

spaceflight_predictions_source = PostgreSQLSource(
    name="spaceflight_predictions_source",
    table="spaceflight_prediction_table",
    timestamp_field="event_timestamp",
)

predictions_features_view = FeatureView(
    name="spaceflight_predictions_features_view",
    entities=[shuttle, company],
    ttl=None,
    schema=[
        Field(name="prediction", dtype=Float64, tags={"type": "batch_scoring_predictions"}),
    ],
    online=True,
    source=spaceflight_predictions_source,
)


shuttle_activity_service = FeatureService(
    name="spaceflight_feature_service_v1",
    features=[shuttle_features_view],
    tags={"description": "Servizio per il monitoraggio flotta e rating company"}
)


spaceflight_evidently = FeatureService(
    name="spaceflight_evidently_feature_view",
    features=[predictions_features_view],
    tags={"description": "Servizio per il monitoraggio flotta e rating company"}
)

#fs = FeatureStore(repo_path="../training-pipeline")
#fs.apply([shuttle, company, shuttle_features_view, shuttle_activity_service])
