import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
import os
from sqlalchemy import create_engine
from feast import FeatureStore

MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_CURRENT_OBJ = "current_target"

def upload_to_minio(df, object_name):
    url = f"s3://datasets/{object_name}.csv"
    df.to_csv(
        url,
        index=False,
        storage_options={
            "key": MINIO_ROOT_USER,
            "secret": MINIO_ROOT_PASSWORD,
            "client_kwargs": {"endpoint_url": MINIO_URL}
        }
    )

st.title("ML Feature Generator")
st.write("Configura rfeatures range and generate new samples")


col1, col2 = st.columns([2, 1])
with col1:
    num_samples = st.number_input(
        "Number of samples",
        min_value=1,
        max_value=10000,
        value=1000,
        step=1
    )

st.subheader("Range delle Features Numeriche")

col1, col2 = st.columns(2)

with col1:
    engines_range = st.slider(
        "Engines - Range",
        min_value=0.0,
        max_value=10.0,
        value=(1.0, 4.0),
        step=1.0
    )

    passenger_capacity_range = st.slider(
        "Passenger Capacity - Range",
        min_value=1,
        max_value=50,
        value=(2, 10),
        step=1
    )

    crew_range = st.slider(
        "Crew - Range",
        min_value=0.0,
        max_value=20.0,
        value=(1.0, 5.0),
        step=1.0
    )

with col2:
    company_rating_range = st.slider(
        "Company Rating - Range",
        min_value=0.0,
        max_value=5.0,
        value=(3.0, 5.0),
        step=0.1
    )

    review_scores_rating_range = st.slider(
        "Review Scores Rating - Range",
        min_value=0.0,
        max_value=100.0,
        value=(3.5, 5.0),
        step=0.1
    )

    price_range = st.slider(
        "Price - Range",
        min_value=0.0,
        max_value=10000.0,
        value=(500.0, 3000.0),
        step=50.0
    )

st.subheader("Boolean features probability")
st.write("Set probability (0-100%) of True values")

col1, col2, col3 = st.columns(3)

with col1:
    d_check_prob = st.slider(
        "D Check Complete (%)",
        min_value=0,
        max_value=100,
        value=70,
        step=1
    )

with col2:
    moon_clearance_prob = st.slider(
        "Moon Clearance Complete (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=1
    )

with col3:
    iata_approved_prob = st.slider(
        "IATA Approved (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=1
    )

st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    generate_button = st.button("Generate new saples", type="primary", use_container_width=True)

if generate_button:
    try:
        if not os.path.exists('data/reference.csv'):
            st.error("File 'reference.csv' not found.")
        else:
            reference_df = pd.read_csv('data/reference.csv')

            if len(reference_df) < num_samples:
                st.warning(
                    f"Il reference.csv ha solo {len(reference_df)} righe, ma hai richiesto {num_samples} campioni. Verranno usate tutte le righe disponibili.")
                current_df = reference_df.copy()
            else:
                current_df = reference_df.sample(n=num_samples, replace=False).reset_index(drop=True)

            generation_timestamp = datetime.now(tz=timezone.utc)
            current_df['engines'] = np.random.randint(int(engines_range[0]), int(engines_range[1]) + 1,
                                                      len(current_df)).astype(float)
            current_df['passenger_capacity'] = np.random.randint(passenger_capacity_range[0],
                                                                 passenger_capacity_range[1] + 1, len(current_df))
            current_df['crew'] = np.random.uniform(crew_range[0], crew_range[1], len(current_df)).round(1)
            current_df['d_check_complete'] = np.random.random(len(current_df)) < (d_check_prob / 100)
            current_df['moon_clearance_complete'] = np.random.random(len(current_df)) < (moon_clearance_prob / 100)
            current_df['iata_approved'] = np.random.random(len(current_df)) < (iata_approved_prob / 100)
            current_df['company_rating'] = np.random.uniform(company_rating_range[0], company_rating_range[1],
                                                             len(current_df)).round(1)
            current_df['review_scores_rating'] = np.random.uniform(review_scores_rating_range[0],
                                                                   review_scores_rating_range[1],
                                                                   len(current_df)).round(1)
            current_df['price'] = np.random.uniform(price_range[0], price_range[1], len(current_df)).round(1)
            current_df['event_timestamp'] = generation_timestamp

            st.session_state.generated_data = current_df
            st.session_state.generated = True
            st.session_state.postgres_success = False
            st.session_state.postgres_error = None
            #st.session_state.api_response = None
            #st.session_state.api_error = None

            try:
                db_host = os.getenv('POSTGRES_HOST', 'postgres')
                db_port = os.getenv('POSTGRES_PORT', '5432')
                db_user = os.getenv('POSTGRES_USER', 'user')
                db_password = os.getenv('POSTGRES_PASSWORD', 'password')
                db_name = os.getenv('POSTGRES_DB', 'spaceflight_db')
                connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
                engine = create_engine(connection_string)

                current_df.to_sql('spaceflight_table', engine, if_exists='append', index=False)

                st.session_state.postgres_success = True
                # Recupera start_date dall'ultimo saved dataset di Feast
                try:
                    feast_repo_path = os.getenv('FEAST_REPO_PATH', '.')
                    store = FeatureStore(repo_path=feast_repo_path)

                    dataset_list = store.list_saved_datasets()
                    valid_datasets = [
                        ds for ds in dataset_list
                        if ds.tags.get('start_date') and ds.tags.get('end_date')
                    ]

                    if valid_datasets:
                        latest_ds = max(
                            valid_datasets,
                            key=lambda x: pd.to_datetime(x.tags.get('end_date'), utc=True)  # aggiunto utc=True
                        )
                        feast_start_date = latest_ds.tags.get('start_date')
                    else:
                        feast_start_date = "2025-01-01T00:00:00+00:00"  # aggiunto offset

                    st.session_state.feast_start_date = feast_start_date

                except Exception as e:
                    st.session_state.feast_start_date = "2025-01-01T00:00:00+00:00"  # aggiunto offset
                upload_to_minio(current_df, MINIO_CURRENT_OBJ + generation_timestamp.strftime("%Y-%m-%d_%H-%M-%S"))


            except Exception as e:
                st.session_state.postgres_error = str(e)

    except Exception as e:
        st.error(f"Errore durante la generazione: {str(e)}")

if 'generated' in st.session_state and st.session_state.generated:
    df = st.session_state.generated_data

    st.markdown("---")
    st.subheader("Status Caricamento Dati")

    if st.session_state.get('postgres_success', False):
        st.success(f"{len(df)} samples loaded on the Offline Store")
    elif st.session_state.get('postgres_error'):
        st.error(f"Errror during loading on the Offline Store: {st.session_state.postgres_error}")

    st.subheader("API call")
    st.write("Run this command to start batch-scoring:")

    start_date = st.session_state.get('feast_start_date', '2025-01-01T00:00:00+00:00')
    end_date = generation_timestamp.isoformat()

    # Normalizza start_date per garantire che sia sempre aware
    start_date = pd.to_datetime(start_date, utc=True).isoformat()

    curl_command = f'''curl -X POST http://localhost:3000/batch-scoring \\
         -H "Content-Type: application/json" \\
         -d '{{
               "request_start_date": "{start_date}",
               "request_end_date": "{end_date}"
             }}' '''

    st.code(curl_command, language="bash")