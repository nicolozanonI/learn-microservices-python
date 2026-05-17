import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
import os
import calendar
from sqlalchemy import create_engine
from feast import FeatureStore
from utils.evidently_integration import (
    project_setup,
    data_drift_check,
    model_performance_check,
)
import subprocess
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ML Feature Generator", layout="wide")  # wide usa tutta la larghezza [3](https://github.com/streamlit/streamlit/issues/6336)[4](https://dev.to/jamesbmour/streamlit-part-6-mastering-layouts-4hci)
st.markdown("""
<style>
/* riduce i margini laterali del contenuto principale */
div.block-container { padding-left: 1.2rem; padding-right: 1.2rem; padding-top: 1.0rem; }
/* riduce un po' lo spazio verticale tra elementi */
div[data-testid="stVerticalBlock"] > div { gap: 0.65rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* CONTENITORE */
.st-key-action_keypad {
  padding-top: 0.25rem;
}

/* DIMENSIONE BOTTONI (questo è già ok) */
.st-key-pad_analyze button,
.st-key-pad_generate button,
.st-key-pad_retrain button,
.st-key-pad_api button,
.st-key-pad_cancel button {
  width: 180px !important;
  height: 40px !important;
  padding: 0 !important;
  border-radius: 12px !important;
  font-size: 0.82rem !important;
  font-weight: 650 !important;
  line-height: 1.05rem !important;
  white-space: nowrap !important;
}


/* ✅ ✅ ✅ COLORI (AGGIUNGI QUESTA PARTE) */

/* Generate (blu) */
.st-key-pad_generate button {
  background-color: #1976D2 !important;
  color: white !important;
}

/* Analyze (viola) */
.st-key-pad_analyze button {
  background-color: #7B1FA2 !important;
  color: white !important;
}

/* Retrain (verde) */
.st-key-pad_retrain button {
  background-color: #2E7D32 !important;
  color: white !important;
}

/* API (arancione) */
.st-key-pad_api button {
  background-color: #F57C00 !important;
  color: white !important;
}

/* Cancel (rosso) */
.st-key-pad_cancel button {
  background-color: #D32F2F !important;
  color: white !important;
}


/* CENTRATURA */
.st-key-pad_analyze button > div,
.st-key-pad_generate button > div,
.st-key-pad_retrain button > div,
.st-key-pad_api button > div,
.st-key-pad_cancel button > div {
  justify-content: center !important;
}


/* HOVER migliorato */
.st-key-pad_analyze button:hover,
.st-key-pad_generate button:hover,
.st-key-pad_retrain button:hover,
.st-key-pad_api button:hover,
.st-key-pad_cancel button:hover {
  transform: translateY(-1px);
  filter: brightness(1.1);
}


/* DISABLED (importantissimo) */
.st-key-pad_generate button:disabled,
.st-key-pad_analyze button:disabled,
.st-key-pad_retrain button:disabled,
.st-key-pad_api button:disabled,
.st-key-pad_cancel button:disabled {
  background-color: #eeeeee !important;
  color: #9e9e9e !important;
  border: 1px solid #cccccc !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- ENV / CONST ----------------
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_CURRENT_OBJ = "current_target"
KEDRO_API_URL = os.getenv("KEDRO_API_URL", "http://training-pipeline:8005")

MONTHS_IT = [
    ("Gen", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4),
    ("Mag", 5), ("Giu", 6), ("Lug", 7), ("Ago", 8),
    ("Set", 9), ("Ott", 10), ("Nov", 11), ("Dic", 12)
]
NUM_TO_LABEL = {num: lab for lab, num in MONTHS_IT}


# ---------------- STATE + UI HELPERS ----------------
def _init_timeline_state():
    if "selected_months" not in st.session_state:
        st.session_state.selected_months = [datetime.now().month]
    if "months_with_samples" not in st.session_state:
        st.session_state.months_with_samples = set()
    if "month_samples_meta" not in st.session_state:
        st.session_state.month_samples_meta = {}  # {month: {"last_ts": iso, "count": n}}
    if "generated" not in st.session_state:
        st.session_state.generated = False
    if "generated_data" not in st.session_state:
        st.session_state.generated_data = None
    if "generated_start_date" not in st.session_state:
        st.session_state.generated_start_date = None
    if "generated_end_date" not in st.session_state:
        st.session_state.generated_end_date = None
    if "feast_start_date" not in st.session_state:
        st.session_state.feast_start_date = "2025-01-01T00:00:00+00:00"
    if "analyze_pending" not in st.session_state:
        st.session_state.analyze_pending = False
    if "analyze_reference_month" not in st.session_state:
        st.session_state.analyze_reference_month = None
    if "analyze_current_month" not in st.session_state:
        st.session_state.analyze_current_month = None
    if "analyze_results" not in st.session_state:
        st.session_state.analyze_results = None
    if "analyze_last_pair" not in st.session_state:
        st.session_state.analyze_last_pair = None
    if "feast_apply_running" not in st.session_state:
        st.session_state.feast_apply_running = False
    if "feast_apply_last" not in st.session_state:
        st.session_state.feast_apply_last = None



def toggle_month(month_num: int):
    current = st.session_state.get("selected_months", [])

    if month_num in current:
        current.remove(month_num)
    else:
        current.append(month_num)

    current = sorted(current)

    if not are_consecutive(current) and len(current) > 1:
        current = [month_num]

    st.session_state.selected_months = current

def month_bounds_utc(year: int, month: int):
    last_day = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return start, end

def are_consecutive(months: list[int]) -> bool:
    if not months:
        return False
    months_sorted = sorted(months)
    return all(b - a == 1 for a, b in zip(months_sorted, months_sorted[1:]))

def multi_month_bounds(year: int, months: list[int]):
    months_sorted = sorted(months)

    start_month = months_sorted[0]
    end_month = months_sorted[-1]

    start = datetime(year, start_month, 1, tzinfo=timezone.utc)

    last_day = calendar.monthrange(year, end_month)[1]
    end = datetime(year, end_month, last_day, 23, 59, 59, tzinfo=timezone.utc)

    return start, end
def inject_month_css(months_with_samples: set, selected_months: list):
    css = ["<style>"]

    # ─── LINEA: background-image sul container flex delle colonne ───────────
    # Targeting via key "timeline_row" → niente z-index battle
    css.append("""
    .st-key-timeline_row [data-testid="stHorizontalBlock"] {
        background-image: linear-gradient(
            to bottom,
            transparent         calc(50% - 2px),
            #e0e0e0             calc(50% - 2px),
            #9e9e9e             50%,
            #e0e0e0             calc(50% + 2px),
            transparent         calc(50% + 2px)
        );
        background-repeat: no-repeat;
        background-size: 95% 100%;       /* ← larghezza linea */
        background-position: 4.5% 0;    /* ← offset da sinistra */
        padding: 1rem 0;
    }
    """)

    for _, m in MONTHS_IT:
        key = f"m{m:02d}"
        sel = f".st-key-{key} button"

        # Stile base del nodo
        css.append(f"""{sel} {{
            background-color: {'#B3E5FC' if m in months_with_samples else 'white'} !important;
            color: {'#0D47A1' if m in months_with_samples else '#424242'} !important;
            border: 2px solid {'#29B6F6' if m in months_with_samples else '#bdbdbd'} !important;
            border-radius: 50% !important;
            width: 44px !important;
            height: 44px !important;
            min-width: 44px !important;
            min-height: 44px !important;
            padding: 0 !important;
            font-weight: 600;
            font-size: 0.9rem !important;
            transition: all 0.25s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.08);
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}""")

        # Hover
        css.append(f"""{sel}:hover {{
            transform: translateY(-2px) scale(1.08);
            border-color: {'#0288D1' if m in months_with_samples else '#616161'} !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
            color: {'#01579B' if m in months_with_samples else '#212121'} !important;
        }}""")

        # Selected
        if m in selected_months:
            if m in months_with_samples:
                css.append(f"""{sel} {{
                            background-color: #B3E5FC !important;
                            color: #01579B !important;
                            border: 3px solid #1976D2 !important;
                            box-shadow:
                                0 0 0 4px rgba(25,118,210,0.30),
                                0 4px 12px rgba(25,118,210,0.35) !important;
                            transform: scale(1.12);
                            font-weight: 700 !important;
                        }}""")
                css.append(f"""{sel}:hover {{
                            transform: translateY(-2px) scale(1.12);
                            box-shadow:
                                0 0 0 5px rgba(25,118,210,0.40),
                                0 6px 20px rgba(25,118,210,0.45) !important;
                        }}""")
            else:
                # Selezionato ma non ancora generato → sfondo bianco con anello
                css.append(f"""{sel} {{
                            background-color: white !important;
                            color: #1976D2 !important;
                            border: 3px solid #1976D2 !important;
                            box-shadow:
                                0 0 0 4px rgba(25,118,210,0.25),
                                0 4px 12px rgba(25,118,210,0.35) !important;
                            transform: scale(1.12);
                            font-weight: 700 !important;
                        }}""")
                css.append(f"""{sel}:hover {{
                            transform: translateY(-2px) scale(1.12);
                            box-shadow:
                                0 0 0 5px rgba(25,118,210,0.35),
                                0 6px 20px rgba(25,118,210,0.45) !important;
                        }}""")

    css.append("</style>")
    st.markdown("\n".join(css), unsafe_allow_html=True)

def run_feast_apply():

    st.session_state.feast_apply_running = True
    st.session_state.feast_apply_last = None

    repo_dir = Path(__file__).resolve().parent

    try:
        proc = subprocess.run(
            ["feast", "-c", str(repo_dir), "apply"],
            capture_output=True,
            text=True,
        )
        st.session_state.feast_apply_last = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "repo_dir": str(repo_dir),
            "command": f"feast -c {repo_dir} apply",
        }
    except FileNotFoundError as e:
        st.session_state.feast_apply_last = {
            "returncode": -1,
            "stdout": "",
            "stderr": "Comando `feast` non trovato nel container. Assicurati che feast sia installato nell'immagine.",
            "repo_dir": str(repo_dir),
            "command": "feast ...",
        }
    except Exception as e:
        st.session_state.feast_apply_last = {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Errore durante feast apply: {e}",
            "repo_dir": str(repo_dir),
            "command": f"feast -c {repo_dir} apply",
        }
    finally:
        st.session_state.feast_apply_running = False
        st.rerun()

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


def postgres_engine():
    db_host = os.getenv('POSTGRES_HOST', 'postgres')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_user = os.getenv('POSTGRES_USER', 'user')
    db_password = os.getenv('POSTGRES_PASSWORD', 'password')
    db_name = os.getenv('POSTGRES_DB', 'spaceflight_db')
    connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    return create_engine(connection_string)


# ---------------- UI ----------------
_init_timeline_state()

st.title("ML Feature Generator")

st.markdown("**Create or update a feature store deployment:**")

st.button(
    "🧩 Feast apply",
    key="btn_feast_apply",
    disabled=st.session_state.get("feast_apply_running", False),
    on_click=run_feast_apply,
)

st.markdown("---")

st.write("Configura features range and generate new samples")

top_left, top_right = st.columns([2, 1])
with top_left:
    num_samples = st.number_input("Number of samples", min_value=1, value=10000, step=1)


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
    d_check_prob = st.slider("D Check Complete (%)", 0, 100, 70, 1)
    moon_clearance_prob = st.slider("Moon Clearance Complete (%)", 0, 100, 60, 1)
    iata_approved_prob = st.slider("IATA Approved (%)", 0, 100, 50, 1)

st.markdown("---")

months_with_samples = st.session_state.get("months_with_samples", set())
selected_months = st.session_state.get("selected_months", [])

# 2 colonne: timeline larga + keypad stretta
left, right = st.columns([7.2, 2.8], gap="small", vertical_alignment="top")

with left:
    st.subheader("Timeline (1 anno / 12 mesi)")
    inject_month_css(months_with_samples, selected_months)

    timeline = st.container(key="timeline_row")
    with timeline:
        mcols = st.columns(12, gap="small")
        for i, (lab, m) in enumerate(MONTHS_IT):
            with mcols[i]:
                st.button(
                    lab,
                    key=f"m{m:02d}",
                    on_click=toggle_month,
                    args=(m,),
                    use_container_width=True,
                )

with right:
    # --- REGOLE BOTTONI (stesse tue) ---
    pending = st.session_state.get("analyze_pending", False)
    ref_month = st.session_state.get("analyze_reference_month")

    generate_enabled = (len(selected_months) >= 1 and are_consecutive(selected_months))
    analyze_enabled  = ((not pending) and (len(selected_months) == 1) and set(selected_months).issubset(months_with_samples))
    retrain_enabled  = (are_consecutive(selected_months) and set(selected_months).issubset(months_with_samples))
    api_call_enabled = (are_consecutive(selected_months) and set(selected_months).issubset(months_with_samples))

    # --- KEYpad 2x2 + 1 ---
    keypad = st.container(key="action_keypad")
    with keypad:
        # Riga 1
        r1c1, r1c2 = st.columns(2, gap="xsmall")
        with r1c1:
            generate_button = st.button("✨\nGenerate", key="pad_generate", type="primary", disabled=not generate_enabled, width="content")  #
        with r1c2:
            analyze_button = st.button("🔎\nAnalyze", key="pad_analyze", disabled=not analyze_enabled, width="content")

        r2c1, r2c2 = st.columns(2, gap="xsmall")
        with r2c1:
            retrain_button = st.button("️🔁\nTrain/Retrain", key="pad_retrain", disabled=not retrain_enabled, width="content")  #
        with r2c2:
            cancel_analyze = st.button("✖️\nCancel", key="pad_cancel", disabled=not pending, width="content")  #

        r3c1, r3c2 = st.columns(2, gap="xsmall")
        with r3c1:
            api_call_button = st.button("📄\nAPI", key="pad_api", disabled=not api_call_enabled, width="content")
        with r3c2:
            st.write("")

    # --- CLICK HANDLERS (identici ai tuoi) ---
    if analyze_button:
        ref = int(selected_months[0])
        st.session_state.analyze_pending = True
        st.session_state.analyze_reference_month = ref
        st.session_state.analyze_current_month = None
        st.session_state.analyze_last_pair = None
        st.session_state.selected_months = []
        st.rerun()

    if cancel_analyze and pending:
        st.session_state.analyze_pending = False
        st.session_state.analyze_reference_month = None
        st.session_state.analyze_current_month = None
        st.rerun()

    # --- HINT (sotto la tastierina) ---
    msg = ""
    if pending and ref_month is not None:
        msg = f"Step 2: reference = {NUM_TO_LABEL.get(ref_month, ref_month)} → scegli current (azzurro)."
    else:
        if not generate_enabled:
            msg = "Generate: seleziona mesi consecutivi."
        elif not analyze_enabled:
            msg = "Analyze: seleziona 1 mese già generato."
        elif not retrain_enabled:
            msg = "Train/Retrain: seleziona mesi consecutivi già generati."
        elif not api_call_enabled:
            msg = "API: seleziona mesi consecutivi già generati."
    st.caption(msg if msg else " ")
if "btn_apicall" in st.session_state:
    pass

if 'btn_apicall' not in st.session_state:
    st.session_state.btn_apicall = False

if api_call_button:
    year = datetime.now(timezone.utc).year

    m_start, m_end = multi_month_bounds(year, selected_months)

    labels = [NUM_TO_LABEL[m] for m in selected_months]

    curl_command = f'''curl -X POST http://localhost:3000/batch-scoring \\
  -H "Content-Type: application/json" \\
  -d '{{
        "start_date": "{m_start.isoformat()}",
        "end_date": "{m_end.isoformat()}"
      }}' '''

    st.subheader(f"API call ({', '.join(labels)})")

    st.code(curl_command, language="bash")
def _simplify_failed_tests(failed_tests):
    out = []
    for t in failed_tests or []:
        try:
            out.append({
                "id": getattr(t, "id", None),
                "name": getattr(t, "name", None),
                "column": getattr(getattr(t, "metric_config", None), "params", {}).get("column")
                          if getattr(t, "metric_config", None) else None,
            })
        except Exception:
            out.append({"name": str(t)})
    return out


# ---------------- LOGICA: ANALYZE STEP 2 (Evidently) ----------------
pending = st.session_state.get("analyze_pending", False)
ref_month = st.session_state.get("analyze_reference_month")
selected_months = st.session_state.get("selected_months", [])
months_with_samples = st.session_state.get("months_with_samples", set())

if pending and ref_month is not None and len(selected_months) == 1:
    cur_month = int(selected_months[0])

    if cur_month == int(ref_month):
        st.warning("Analyze: scegli un mese current diverso dal reference.")
    elif cur_month not in months_with_samples:
        st.warning("Analyze: il mese current deve essere già generato (azzurro).")
    else:
        pair = (int(ref_month), int(cur_month))

        if st.session_state.get("analyze_last_pair") != pair:
            st.session_state.analyze_last_pair = pair

            year = datetime.now(timezone.utc).year
            ref_start, ref_end = month_bounds_utc(year, int(ref_month))
            cur_start, cur_end = month_bounds_utc(year, int(cur_month))

            with st.spinner("Evidently analysis in corso..."):
                try:
                    project = project_setup()

                    failed_data = data_drift_check(project, ref_start, ref_end, cur_start, cur_end)
                    failed_model = model_performance_check(project, ref_start, ref_end, cur_start, cur_end)

                    st.session_state.analyze_results = {
                        "status": "OK",
                        "reference_month": int(ref_month),
                        "current_month": int(cur_month),
                        "failed_data_tests_count": len(failed_data),
                        "failed_model_tests_count": len(failed_model),
                        "failed_data_tests": _simplify_failed_tests(failed_data),
                        "failed_model_tests": _simplify_failed_tests(failed_model),
                        "evidently_project_id": getattr(project, "id", None),
                    }
                except Exception as e:
                    st.session_state.analyze_results = {
                        "status": "ERROR",
                        "error": str(e),
                        "reference_month": int(ref_month),
                        "current_month": int(cur_month),
                    }

            st.session_state.analyze_pending = False
            st.session_state.analyze_reference_month = None
            st.session_state.analyze_current_month = None
            st.session_state.selected_months = []
            st.rerun()

# ---------------- LOGICA: GENERATE (1 mese) ----------------
if generate_button:
    try:
        selected_months = st.session_state.get("selected_months", [])
        if not are_consecutive(selected_months):
            st.error("Generate richiede ESATTAMENTE 1 mese selezionato.")
            st.stop()

        if not os.path.exists('data/reference.csv'):
            st.error("File 'reference.csv' not found.")
            st.stop()

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

        year = datetime.now(timezone.utc).year
        month_start, month_end = multi_month_bounds(year, selected_months)

        start_u = int(month_start.timestamp())
        end_u = int(month_end.timestamp())

        rand_u = np.random.randint(start_u, end_u + 1, size=len(current_df), dtype=np.int64)
        current_df["event_timestamp"] = pd.to_datetime(rand_u, unit="s", utc=True)

        # Start/end per curl batch scoring: mese intero
        st.session_state.generated_start_date = month_start.isoformat()
        st.session_state.generated_end_date = month_end.isoformat()

        st.session_state.generated_data = current_df
        st.session_state.generated = True
        st.session_state.postgres_success = False
        st.session_state.postgres_error = None

        st.session_state.months_with_samples.update(selected_months)

        for m in selected_months:
            st.session_state.month_samples_meta[m] = {
                "last_ts": generation_timestamp.isoformat(),
                "count": int(num_samples),
            }

        try:
            engine = postgres_engine()
            current_df.to_sql('spaceflight_table', engine, if_exists='append', index=False)
            st.session_state.postgres_success = True

            try:
                feast_repo_path = os.getenv('FEAST_REPO_PATH', '.')
                store = FeatureStore(repo_path=feast_repo_path)
                dataset_list = store.list_saved_datasets()
                valid_datasets = [ds for ds in dataset_list if ds.tags.get('start_date') and ds.tags.get('end_date')]

                if valid_datasets:
                    latest_ds = max(valid_datasets, key=lambda x: pd.to_datetime(x.tags.get('end_date'), utc=True))
                    st.session_state.feast_start_date = latest_ds.tags.get('start_date')
                else:
                    st.session_state.feast_start_date = "2025-01-01T00:00:00+00:00"
            except Exception:
                st.session_state.feast_start_date = "2025-01-01T00:00:00+00:00"

            upload_to_minio(current_df, MINIO_CURRENT_OBJ + generation_timestamp.strftime("%Y-%m-%d_%H-%M-%S"))
            st.rerun()

        except Exception as e:
            st.session_state.postgres_error = str(e)

    except Exception as e:
        st.error(f"Errore durante la generazione: {str(e)}")


# ---------------- LOGICA: TRAIN/RETRAIN (1 mese, già generato) ----------------
if retrain_button:
    import time
    from urllib.parse import urljoin

    selected_months = st.session_state.get("selected_months", [])
    months_with_samples = st.session_state.get("months_with_samples", set())

    if not (are_consecutive(selected_months) and set(selected_months).issubset(months_with_samples)):
        st.error("Train/retrain richiede ESATTAMENTE 1 mese selezionato e già generato (azzurro).")
        st.stop()

    retrain_url = os.getenv("RETRAIN_URL", "http://training-pipeline:8005/run-pipeline").strip()
    if not retrain_url:
        st.info("RETRAIN_URL non configurata. Imposta l'env var per abilitare il trigger train/retrain.")
        st.stop()

    year = datetime.now(timezone.utc).year

    month_start, month_end = multi_month_bounds(
        datetime.now(timezone.utc).year,
        selected_months
    )

    payload = {
        "start_date": month_start.isoformat(),
        "end_date": month_end.isoformat(),
        "pipeline": "training",
        "year": year,
        "months": selected_months,
        "num_samples": int(num_samples),
    }

    poll_interval_sec = 2
    max_polls = 1800

    with st.spinner("Training/retraining in corso..."):
        try:
            submit = requests.post(retrain_url, json=payload, timeout=30)

            if submit.status_code not in (200, 202):
                st.error(f"Submit train/retrain fallito: {submit.status_code}")
                st.write(submit.text)
                st.stop()

            submit_json = {}
            try:
                submit_json = submit.json()
            except Exception:
                st.error("La risposta del submit non è JSON valido.")
                st.write(submit.text)
                st.stop()

            job_id = submit_json.get("job_id")
            status_path = submit_json.get("status_url")  # es. "/run-pipeline/<job_id>"

            if not job_id:
                st.error("Submit OK ma manca 'job_id' nella response.")
                st.json(submit_json)
                st.stop()

            if status_path:
                status_url = urljoin(retrain_url, status_path)
            else:
                status_url = urljoin(retrain_url, f"/run-pipeline/{job_id}")

            st.session_state.retrain_job_id = job_id
            st.session_state.retrain_status_url = status_url

            for _ in range(max_polls):
                r = requests.get(status_url, timeout=30)

                if not r.ok:
                    st.error(f"Errore polling status: {r.status_code}")
                    st.write(r.text)
                    st.stop()

                status_json = r.json()
                state = status_json.get("status")

                if state == "completed":
                    st.success("Training/retrain completato ✅")
                    st.json(status_json)
                    break

                if state == "failed":
                    st.error("Training/retrain fallito ❌")
                    err = status_json.get("error")
                    if err:
                        st.subheader("Dettagli errore")
                        st.json(err)
                    else:
                        st.json(status_json)
                    break

                time.sleep(poll_interval_sec)
            else:
                st.warning("Training/retrain ancora in esecuzione (timeout polling). Riprova a controllare più tardi.")
                st.write({"job_id": job_id, "status_url": status_url})

        except Exception as e:
            st.error(f"Errore chiamata training/retrain: {e}")

# ---------------- UI POST-GENERATE ----------------
if st.session_state.get("generated", False):
    df_last = st.session_state.generated_data
    st.markdown("---")

    if st.session_state.get('postgres_success', False):
        st.success(f"{len(df_last)} samples loaded on the Offline Store")
    elif st.session_state.get('postgres_error'):
        st.error(f"Error during loading on the Offline Store: {st.session_state.postgres_error}")

    st.subheader("API call")
    st.write("Run this command to start batch-scoring:")

    start_date = st.session_state.get("generated_start_date")
    end_date = st.session_state.get("generated_end_date")

    if not start_date:
        start_date = st.session_state.get('feast_start_date', '2025-01-01T00:00:00+00:00')
        start_date = pd.to_datetime(start_date, utc=True).isoformat()

    if not end_date:
        end_date = datetime.now(timezone.utc).isoformat()

    curl_command = f'''curl -X POST http://localhost:3000/batch-scoring \\
  -H "Content-Type: application/json" \\
  -d '{{
        "start_date": "{start_date}",
        "end_date": "{end_date}"
      }}' '''

    st.code(curl_command, language="bash")

# ---------------- UI POST-ANALYZE ----------------
if st.session_state.get("analyze_results") is not None:
    res = st.session_state.analyze_results
    st.markdown("---")
    st.subheader("Analyze / Evidently Results")

    if res.get("status") == "OK":
        ref_m = res["reference_month"]
        cur_m = res["current_month"]
        st.write({
            "reference": NUM_TO_LABEL.get(ref_m, ref_m),
            "current": NUM_TO_LABEL.get(cur_m, cur_m),
            "failed_data_tests": res.get("failed_data_tests_count", 0),
            "failed_model_tests": res.get("failed_model_tests_count", 0),
            "evidently_project_id": res.get("evidently_project_id"),
        })

        if res.get("failed_data_tests_count", 0) > 0:
            st.markdown("**Failed Data Drift Tests**")
            st.dataframe(pd.DataFrame(res.get("failed_data_tests", [])))

        if res.get("failed_model_tests_count", 0) > 0:
            st.markdown("**Failed Model Performance Tests**")
            st.dataframe(pd.DataFrame(res.get("failed_model_tests", [])))
    else:
        st.error("Evidently analysis fallita")
        st.write(res)

    # ---------------- UI: FEAST APPLY OUTPUT ----------------
    if st.session_state.get("feast_apply_running", False):
        with st.spinner("Eseguo `feast apply`..."):
            st.write("In corso...")
    elif st.session_state.get("feast_apply_last") is not None:
        res = st.session_state.feast_apply_last
        st.markdown("---")
        st.subheader("Feast apply output")

        st.caption(f"Repo: {res.get('repo_dir')}")
        st.code(res.get("command", ""), language="bash")

        if res.get("returncode", 1) == 0:
            st.success("feast apply completato ✅")
        else:
            st.error(f"feast apply fallito ❌ (return code: {res.get('returncode')})")

        if res.get("stdout"):
            st.markdown("**stdout**")
            st.code(res["stdout"], language="text")

        if res.get("stderr"):
            st.markdown("**stderr**")
            st.code(res["stderr"], language="text")