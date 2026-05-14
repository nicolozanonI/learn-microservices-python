import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
import os
from sqlalchemy import create_engine
from feast import FeatureStore

# --- Pagina WIDE + meno padding laterale (più spazio utile) ---
st.set_page_config(page_title="ML Feature Generator", layout="wide")  # wide usa tutta la larghezza [1](https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config)[2](https://stackoverflow.com/questions/78371971/removing-the-wide-margins-in-a-streamlit-app)
st.markdown("""
<style>
/* riduce i margini laterali del contenuto principale */
div.block-container { padding-left: 1.2rem; padding-right: 1.2rem; padding-top: 1.0rem; }
/* riduce un po' lo spazio verticale tra elementi */
div[data-testid="stVerticalBlock"] > div { gap: 0.65rem; }
</style>
""", unsafe_allow_html=True)

MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_CURRENT_OBJ = "current_target"

# --- Timeline mesi (toggle buttons + styling) ---
MONTHS_IT = [
    ("Gen", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4),
    ("Mag", 5), ("Giu", 6), ("Lug", 7), ("Ago", 8),
    ("Set", 9), ("Ott", 10), ("Nov", 11), ("Dic", 12)
]
NUM_TO_LABEL = {num: lab for lab, num in MONTHS_IT}


def _init_timeline_state():
    if "selected_months" not in st.session_state:
        st.session_state.selected_months = [datetime.now().month]  # default: mese corrente
    if "months_with_samples" not in st.session_state:
        st.session_state.months_with_samples = set()
    if "month_samples_meta" not in st.session_state:
        st.session_state.month_samples_meta = {}  # {month: {"last_ts": iso, "count": n}}


def toggle_month(month_num: int):
    sel = set(st.session_state.get("selected_months", []))
    if month_num in sel:
        sel.remove(month_num)
    else:
        sel.add(month_num)
    st.session_state.selected_months = sorted(sel)


def inject_month_css(months_with_samples: set, selected_months: list):
    """
    Azzurro chiaro = mese con samples già generati.
    Outline = mese selezionato.
    """
    css = ["<style>"]
    for _, m in MONTHS_IT:
        key = f"m{m:02d}"
        sel = f".st-key-{key} button"  # key -> class st-key-... [3](https://docs.streamlit.io/develop/api-reference/widgets/st.button)[4](https://medium.com/@jonathan.alles/professional-streamlit-styling-with-css-and-st-yled-e5c470deaf46)

        if m in months_with_samples:
            css.append(f"""{sel} {{
                background-color: #B3E5FC !important;
                color: #0D47A1 !important;
                border: 1px solid #4FC3F7 !important;
                border-radius: 999px !important;
                padding: 0.20rem 0.55rem !important;
                min-height: 2.2rem !important;
            }}""")
            css.append(f"""{sel}:hover {{
                background-color: #81D4FA !important;
                border-color: #29B6F6 !important;
                color: #0D47A1 !important;
            }}""")
        else:
            css.append(f"""{sel} {{
                background-color: transparent !important;
                color: inherit !important;
                border: 1px solid rgba(49,51,63,0.25) !important;
                border-radius: 999px !important;
                padding: 0.20rem 0.55rem !important;
                min-height: 2.2rem !important;
            }}""")
            css.append(f"""{sel}:hover {{
                border-color: rgba(49,51,63,0.55) !important;
            }}""")

        if m in selected_months:
            css.append(f"""{sel} {{
                box-shadow: 0 0 0 0.22rem rgba(30,136,229,.35) !important;
            }}""")

    css.append("</style>")
    st.markdown("\n".join(css), unsafe_allow_html=True)


def inject_action_buttons_css():
    st.markdown("""
    <style>
    .st-key-btn_analyze button,
    .st-key-btn_generate button,
    .st-key-btn_retrain button{
        padding: 0.28rem 0.60rem !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        min-height: 2.3rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


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


# ---------------- UI ----------------
_init_timeline_state()

st.title("ML Feature Generator")
st.write("Configura features range and generate new samples")

# samples input (sopra)
top_left, top_right = st.columns([2, 1])
with top_left:
    num_samples = st.number_input("Number of samples", min_value=1, value=10000, step=1)

# Layout principale: numeriche a sinistra + booleane in colonna a destra
left_panel, right_panel = st.columns([3.2, 1.3], gap="large")

with left_panel:
    st.subheader("Range delle Features Numeriche")
    ncol1, ncol2 = st.columns(2, gap="medium")

    with ncol1:
        engines_range = st.slider("Engines - Range", 0.0, 10.0, (1.0, 4.0), 1.0)
        passenger_capacity_range = st.slider("Passenger Capacity - Range", 1, 50, (2, 10), 1)
        crew_range = st.slider("Crew - Range", 0.0, 20.0, (1.0, 5.0), 1.0)

    with ncol2:
        company_rating_range = st.slider("Company Rating - Range", 0.0, 5.0, (3.0, 5.0), 0.1)
        review_scores_rating_range = st.slider("Review Scores Rating - Range", 0.0, 100.0, (3.5, 35.0), 0.1)
        price_range = st.slider("Price - Range", 0.0, 10000.0, (500.0, 3000.0), 50.0)

with right_panel:
    st.subheader("Boolean features probability")
    st.write("Set probability (0-100%) of True values")

    # una sopra l'altra (come richiesto)
    d_check_prob = st.slider("D Check Complete (%)", 0, 100, 70, 1)
    moon_clearance_prob = st.slider("Moon Clearance Complete (%)", 0, 100, 60, 1)
    iata_approved_prob = st.slider("IATA Approved (%)", 0, 100, 50, 1)

# Riga unica: timeline + bottoni
st.markdown("---")
bottom_left, bottom_right = st.columns([7.5, 2.5], vertical_alignment="center")

with bottom_left:
    st.subheader("Timeline (1 anno / 12 mesi)")
    months_with_samples = st.session_state.months_with_samples
    selected_months = st.session_state.selected_months

    inject_month_css(months_with_samples, selected_months)

    mcols = st.columns(12, gap="small")
    for i, (lab, m) in enumerate(MONTHS_IT):
        with mcols[i]:
            st.button(
                lab,
                key=f"m{m:02d}",
                use_container_width=True,
                on_click=toggle_month,
                args=(m,),
            )

    if selected_months:
        sel_labels = [NUM_TO_LABEL[m] for m in selected_months]
        st.caption(f"Mesi selezionati: {', '.join(sel_labels)}")
    else:
        st.caption("Nessun mese selezionato.")

with bottom_right:
    months_with_samples = st.session_state.get("months_with_samples", set())
    selected_months = st.session_state.get("selected_months", [])

    invalid_selected = set(selected_months) - set(months_with_samples)
    ops_enabled = (len(selected_months) > 0) and (len(invalid_selected) == 0)

    inject_action_buttons_css()

    analyze_button = st.button("Analyze", key="btn_analyze", use_container_width=True, disabled=not ops_enabled)
    generate_button = st.button("Generate", key="btn_generate", type="primary", use_container_width=True)
    retrain_button = st.button("Retrain", key="btn_retrain", use_container_width=True, disabled=not ops_enabled)

    if not ops_enabled and selected_months:
        bad = [NUM_TO_LABEL[m] for m in sorted(invalid_selected)]
        st.caption(f"Analyze/Retrain solo per mesi con samples. Mesi non validi: {', '.join(bad)}")

# ---------------- LOGICA: GENERATE ----------------
if generate_button:
    try:
        if not os.path.exists('data/reference.csv'):
            st.error("File 'reference.csv' not found.")
        else:
            reference_df = pd.read_csv('data/reference.csv')
            current_df = reference_df.sample(n=num_samples, replace=True).reset_index(drop=True)

            generation_timestamp = datetime.now(tz=timezone.utc)
            st.session_state.generation_timestamp = generation_timestamp

            current_df['engines'] = np.random.randint(int(engines_range[0]), int(engines_range[1]) + 1, len(current_df)).astype(float)
            current_df['passenger_capacity'] = np.random.randint(passenger_capacity_range[0], passenger_capacity_range[1] + 1, len(current_df))
            current_df['crew'] = np.random.uniform(crew_range[0], crew_range[1], len(current_df)).round(1)

            current_df['d_check_complete'] = np.random.random(len(current_df)) < (d_check_prob / 100)
            current_df['moon_clearance_complete'] = np.random.random(len(current_df)) < (moon_clearance_prob / 100)
            current_df['iata_approved'] = np.random.random(len(current_df)) < (iata_approved_prob / 100)

            current_df['company_rating'] = np.random.uniform(company_rating_range[0], company_rating_range[1], len(current_df)).round(1)
            current_df['review_scores_rating'] = np.random.uniform(review_scores_rating_range[0], review_scores_rating_range[1], len(current_df)).round(1)
            current_df['price'] = np.random.uniform(price_range[0], price_range[1], len(current_df)).round(1)

            current_df['event_timestamp'] = generation_timestamp

            st.session_state.generated_data = current_df
            st.session_state.generated = True
            st.session_state.postgres_success = False
            st.session_state.postgres_error = None

            # --- Marca i mesi selezionati come "con samples" ---
            sel_months = st.session_state.get("selected_months", [])
            st.session_state.months_with_samples.update(sel_months)

            for m in sel_months:
                st.session_state.month_samples_meta[m] = {
                    "last_ts": generation_timestamp.isoformat(),
                    "count": int(num_samples),
                }

            # --- Offline store (Postgres) ---
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
                    valid_datasets = [ds for ds in dataset_list if ds.tags.get('start_date') and ds.tags.get('end_date')]

                    if valid_datasets:
                        latest_ds = max(valid_datasets, key=lambda x: pd.to_datetime(x.tags.get('end_date'), utc=True))
                        feast_start_date = latest_ds.tags.get('start_date')
                    else:
                        feast_start_date = "2025-01-01T00:00:00+00:00"

                    st.session_state.feast_start_date = feast_start_date
                except Exception:
                    st.session_state.feast_start_date = "2025-01-01T00:00:00+00:00"

                upload_to_minio(current_df, MINIO_CURRENT_OBJ + generation_timestamp.strftime("%Y-%m-%d_%H-%M-%S"))

            except Exception as e:
                st.session_state.postgres_error = str(e)

    except Exception as e:
        st.error(f"Errore durante la generazione: {str(e)}")

# ---------------- LOGICA: ANALYZE ----------------
if analyze_button:
    if not st.session_state.get("generated", False):
        st.warning("Prima genera i dati con 'Generate'.")
    else:
        df = st.session_state.generated_data.copy()
        selected_months = st.session_state.get("selected_months", [])
        selected_months_labels = [NUM_TO_LABEL[m] for m in selected_months]

        st.subheader("Analyze / Risultati analisi")
        st.write({
            "rows": len(df),
            "cols": len(df.columns),
            "selected_months": selected_months_labels,
        })

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        bool_cols = df.select_dtypes(include=[bool]).columns.tolist()

        if num_cols:
            st.markdown("**Statistiche numeriche**")
            st.dataframe(df[num_cols].describe().T)

        if bool_cols:
            st.markdown("**Distribuzioni boolean**")
            bool_summary = pd.DataFrame({
                "true_%": (df[bool_cols].mean() * 100).round(2),
                "true_count": df[bool_cols].sum(),
                "false_count": (~df[bool_cols]).sum()
            })
            st.dataframe(bool_summary)

        st.markdown("**Preview campioni**")
        st.dataframe(df.head(20))

# ---------------- LOGICA: RETRAIN ----------------
if retrain_button:
    retrain_url = os.getenv("RETRAIN_URL", "").strip()
    selected_months_nums = st.session_state.get("selected_months", [])

    if not retrain_url:
        st.info("RETRAIN_URL non configurata. Imposta l'env var per abilitare il trigger retrain.")
    else:
        payload = {
            "year": datetime.now().year,
            "months": selected_months_nums,
            "generation_timestamp": (
                st.session_state.get("generation_timestamp").isoformat()
                if st.session_state.get("generation_timestamp") else None
            ),
            "num_samples": int(num_samples),
        }

        with st.spinner("Trigger retrain in corso..."):
            try:
                r = requests.post(retrain_url, json=payload, timeout=60)
                if r.ok:
                    st.success("Retrain triggerato con successo.")
                    try:
                        st.json(r.json())
                    except Exception:
                        st.write(r.text)
                else:
                    st.error(f"Retrain fallito: {r.status_code}")
                    st.write(r.text)
            except Exception as e:
                st.error(f"Errore chiamata retrain: {e}")

# ---------------- UI POST-GENERATE ----------------
if st.session_state.get("generated", False):
    df = st.session_state.generated_data

    st.markdown("---")

    if st.session_state.get('postgres_success', False):
        st.success(f"{len(df)} samples loaded on the Offline Store")
    elif st.session_state.get('postgres_error'):
        st.error(f"Error during loading on the Offline Store: {st.session_state.postgres_error}")

    st.subheader("API call")
    st.write("Run this command to start batch-scoring:")

    start_date = st.session_state.get('feast_start_date', '2025-01-01T00:00:00+00:00')
    end_date = st.session_state.get('generation_timestamp', datetime.now()).isoformat()
    start_date = pd.to_datetime(start_date, utc=True).isoformat()

    curl_command = f'''curl -X POST http://localhost:3000/batch-scoring \\
         -H "Content-Type: application/json" \\
         -d '{{
               "start_date": "{start_date}",
               "end_date": "{end_date}"
             }}' '''

    st.code(curl_command, language="bash")