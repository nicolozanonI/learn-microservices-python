import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os
from sqlalchemy import create_engine

st.title("🎯 ML Feature Generator")
st.write("Configura i range delle feature e genera campioni per il tuo progetto di machine learning")



# Numero di campioni da generare
col1, col2 = st.columns([2, 1])
with col1:
    num_samples = st.number_input(
        "Numero di campioni da generare",
        min_value=1,
        max_value=10000,
        value=1000,
        step=1
    )

st.subheader("Range delle Features Numeriche")

# Slider disposti in 2 colonne
col1, col2 = st.columns(2)

with col1:
    engines_range = st.slider(
        "Engines - Range",
        min_value=0.0,
        max_value=10.0,
        value=(1.0, 4.0),
        step=0.5
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
        step=0.5
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
        max_value=5.0,
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

st.subheader("Probabilità Features Booleane")
st.write("Imposta la probabilità (0-100%) che il valore sia True")

col1, col2, col3 = st.columns(3)

with col1:
    d_check_prob = st.slider(
        "D Check Complete (%)",
        min_value=0,
        max_value=100,
        value=70,
        step=5
    )

with col2:
    moon_clearance_prob = st.slider(
        "Moon Clearance Complete (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=5
    )

with col3:
    iata_approved_prob = st.slider(
        "IATA Approved (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5
    )

# Pulsante centrato sotto
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    generate_button = st.button("🚀 Generate new saples", type="primary", use_container_width=True)

# Gestione Generate new samples
if generate_button:
    try:
        # Verifica che il file reference.csv esista
        if not os.path.exists('data/reference.csv'):
            st.error("❌ File 'reference.csv' non trovato! Assicurati che esista nella cartella dell'applicazione.")
        else:
            # Carica il reference dataset
            reference_df = pd.read_csv('data/reference.csv')

            # Verifica che ci siano abbastanza righe
            if len(reference_df) < num_samples:
                st.warning(
                    f"⚠️ Il reference.csv ha solo {len(reference_df)} righe, ma hai richiesto {num_samples} campioni. Verranno usate tutte le righe disponibili.")
                sampled_df = reference_df.copy()
            else:
                # Campiona righe casuali
                sampled_df = reference_df.sample(n=num_samples, replace=False).reset_index(drop=True)

            # Timestamp di generazione
            generation_timestamp = datetime.now()

            # Sostituisci le feature generate con i nuovi valori
            sampled_df['engines'] = np.random.randint(int(engines_range[0]), int(engines_range[1]) + 1,
                                                      len(sampled_df)).astype(float)
            sampled_df['passenger_capacity'] = np.random.randint(passenger_capacity_range[0],
                                                                 passenger_capacity_range[1] + 1, len(sampled_df))
            sampled_df['crew'] = np.random.uniform(crew_range[0], crew_range[1], len(sampled_df)).round(1)
            sampled_df['d_check_complete'] = np.random.random(len(sampled_df)) < (d_check_prob / 100)
            sampled_df['moon_clearance_complete'] = np.random.random(len(sampled_df)) < (moon_clearance_prob / 100)
            sampled_df['iata_approved'] = np.random.random(len(sampled_df)) < (iata_approved_prob / 100)
            sampled_df['company_rating'] = np.random.uniform(company_rating_range[0], company_rating_range[1],
                                                             len(sampled_df)).round(1)
            sampled_df['review_scores_rating'] = np.random.uniform(review_scores_rating_range[0],
                                                                   review_scores_rating_range[1],
                                                                   len(sampled_df)).round(1)
            sampled_df['price'] = np.random.uniform(price_range[0], price_range[1], len(sampled_df)).round(1)
            sampled_df['event_timestamp'] = generation_timestamp

            # Salva nel session state per visualizzazione
            st.session_state.generated_data = sampled_df
            st.session_state.generated = True
            st.session_state.postgres_success = False
            st.session_state.postgres_error = None
            st.session_state.api_response = None
            st.session_state.api_error = None

            # Carica su PostgreSQL
            try:
                # Leggi variabili d'ambiente
                db_host = os.getenv('POSTGRES_HOST', 'postgres')
                db_port = os.getenv('POSTGRES_PORT', '5432')
                db_user = os.getenv('POSTGRES_USER', 'user')
                db_password = os.getenv('POSTGRES_PASSWORD', 'password')
                db_name = os.getenv('POSTGRES_DB', 'spaceflight_db')

                # Crea connessione SQLAlchemy
                connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
                engine = create_engine(connection_string)

                # Carica i dati (append, non replace)
                sampled_df.to_sql('spaceflight_table', engine, if_exists='append', index=False)

                st.session_state.postgres_success = True

                # Chiama l'endpoint batch-scoring
                try:
                    api_host = os.getenv('API_HOST', 'inference')
                    api_port = os.getenv('API_PORT', '3000')
                    url = f"http://{api_host}:{api_port}/batch-scoring"

                    # Prepara il payload con il timestamp come stringa
                    payload = {
                        "request": {
                            "request_start_date": "2025-12-01T16:30:45.123456",
                            "request_end_date": generation_timestamp.isoformat()
                        }
                    }

                    headers = {"Content-Type": "application/json"}
                    response = requests.post(url, headers=headers, json=payload, timeout=30)

                    st.session_state.api_response = response

                except Exception as e:
                    st.session_state.api_error = str(e)

            except Exception as e:
                st.session_state.postgres_error = str(e)

    except Exception as e:
        st.error(f"❌ Errore durante la generazione: {str(e)}")

# Visualizzazione dei risultati - solo status
if 'generated' in st.session_state and st.session_state.generated:
    df = st.session_state.generated_data

    st.markdown("---")
    st.subheader("📡 Status Caricamento Dati")

    # PostgreSQL Status
    if st.session_state.get('postgres_success', False):
        st.success(f"✅ {len(df)} campioni caricati con successo su PostgreSQL!")
    elif st.session_state.get('postgres_error'):
        st.error(f"❌ Errore durante il caricamento su PostgreSQL: {st.session_state.postgres_error}")

    # API Response Status
    if st.session_state.get('api_response'):
        response = st.session_state.api_response
        st.success(f"✅ Chiamata a /batch-scoring completata!")
        st.subheader("Risposta API:")

        # Mostra la risposta
        if response.headers.get('content-type') == 'application/json':
            st.json(response.json())
        else:
            st.code(response.text)
    elif st.session_state.get('api_error'):
        st.warning(f"⚠️ Errore durante la chiamata API: {st.session_state.api_error}")