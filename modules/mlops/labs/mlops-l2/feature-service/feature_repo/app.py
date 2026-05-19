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
from urllib.parse import urljoin
import time


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ML Feature Generator", layout="wide")

# ---------------- GLOBAL CSS ----------------
st.markdown(
    """
<style>
/* margini & spacing */
div.block-container { padding-left: 1.2rem; padding-right: 1.2rem; padding-top: 1.0rem; }
div[data-testid="stVerticalBlock"] > div { gap: 0.65rem; }

/* nasconde la decoration/progress bar in alto (la riga giallina) */
div[data-testid="stDecoration"] { display: none !important; }

/* Contenitore keypad */
.st-key-action_keypad { padding-top: 0.25rem; }

/* Dimensione bottoni keypad */
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

/* Colori keypad */
.st-key-pad_generate button { background-color: #1976D2 !important; color: white !important; } /* blu */
.st-key-pad_analyze  button { background-color: #7B1FA2 !important; color: white !important; } /* viola */
.st-key-pad_retrain  button { background-color: #2E7D32 !important; color: white !important; } /* verde */
.st-key-pad_api      button { background-color: #F57C00 !important; color: white !important; } /* arancio */
.st-key-pad_cancel   button { background-color: #D32F2F !important; color: white !important; } /* rosso */

/* Centratura testo */
.st-key-pad_analyze button > div,
.st-key-pad_generate button > div,
.st-key-pad_retrain button > div,
.st-key-pad_api button > div,
.st-key-pad_cancel button > div {
  justify-content: center !important;
}

.st-key-btn_feast_apply button {
  background-color: #FF0000 !important;  /* verde petrolio */
  color: white !important;
  border: 1px solid rgba(0,0,0,0.15) !important;
  border-radius: 12px !important;
}

/* Hover */
.st-key-pad_analyze button:hover,
.st-key-pad_generate button:hover,
.st-key-pad_retrain button:hover,
.st-key-pad_api button:hover,
.st-key-pad_cancel button:hover {
  transform: translateY(-1px);
  filter: brightness(1.1);
}
.st-key-btn_feast_apply button:hover {
  filter: brightness(1.08);
  transform: translateY(-1px);
}

/* Disabled */
.st-key-pad_generate button:disabled,
.st-key-pad_analyze button:disabled,
.st-key-pad_retrain button:disabled,
.st-key-pad_api button:disabled,
.st-key-pad_cancel button:disabled {
  background-color: #eeeeee !important;
  color: #9e9e9e !important;
  border: 1px solid #cccccc !important;
}
.st-key-btn_feast_apply button:disabled {
  background-color: #e0e0e0 !important;
  color: #9e9e9e !important;
  border: 1px solid #cccccc !important;
}


/* Badge esito Feast apply */
.feast-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 650;
  margin-left: 0.5rem;
}
.feast-ok { background: rgba(46,125,50,0.15); color: #2E7D32; border: 1px solid rgba(46,125,50,0.35); }
.feast-ko { background: rgba(211,47,47,0.12); color: #D32F2F; border: 1px solid rgba(211,47,47,0.30); }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- ENV / CONST ----------------
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_CURRENT_OBJ = "current_target"

MONTHS_IT = [
    ("Gen", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4),
    ("Mag", 5), ("Giu", 6), ("Lug", 7), ("Ago", 8),
    ("Set", 9), ("Ott", 10), ("Nov", 11), ("Dic", 12)
]
NUM_TO_LABEL = {num: lab for lab, num in MONTHS_IT}


# ---------------- STATE ----------------
def _init_state():
    if "selected_months" not in st.session_state:
        st.session_state.selected_months = [1]  # default: Gennaio

    if "months_with_samples" not in st.session_state:
        st.session_state.months_with_samples = set()
    if "month_samples_meta" not in st.session_state:
        st.session_state.month_samples_meta = {}

    # generate
    if "generated" not in st.session_state:
        st.session_state.generated = False
    if "generated_data" not in st.session_state:
        st.session_state.generated_data = None
    if "generated_start_date" not in st.session_state:
        st.session_state.generated_start_date = None
    if "generated_end_date" not in st.session_state:
        st.session_state.generated_end_date = None
    if "postgres_success" not in st.session_state:
        st.session_state.postgres_success = False
    if "postgres_error" not in st.session_state:
        st.session_state.postgres_error = None

    # analyze wizard
    if "analyze_pending" not in st.session_state:
        st.session_state.analyze_pending = False
    if "analyze_reference_month" not in st.session_state:
        st.session_state.analyze_reference_month = None
    if "analyze_last_pair" not in st.session_state:
        st.session_state.analyze_last_pair = None
    if "analyze_results" not in st.session_state:
        st.session_state.analyze_results = None

    # feast apply status (solo esito)
    if "feast_apply_running" not in st.session_state:
        st.session_state.feast_apply_running = False
    if "feast_apply_status" not in st.session_state:
        st.session_state.feast_apply_status = None  # "ok" | "ko" | None
    if "feast_apply_last_ts" not in st.session_state:
        st.session_state.feast_apply_last_ts = None

    # action dispatch (evita st.rerun)
    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None
    if "api_call_to_show" not in st.session_state:
        st.session_state.api_call_to_show = None

    # retrain status
    if "retrain_last" not in st.session_state:
        st.session_state.retrain_last = None
    if "retrain_error" not in st.session_state:
        st.session_state.retrain_error = None


def select_single_month(month_num: int):
    """Selezione singola. Blocca selezione current uguale al reference durante Analyze."""

    pending = st.session_state.get("analyze_pending", False)
    ref_month = st.session_state.get("analyze_reference_month")

    if pending and ref_month == month_num:
        return

    current = st.session_state.get("selected_months", [])
    if len(current) == 1 and current[0] == month_num:
        st.session_state.selected_months = []
    else:
        st.session_state.selected_months = [month_num]

    st.session_state.api_call_to_show = None

def clear_selected_month():
    st.session_state.selected_months = []

def month_bounds_utc(year: int, month: int):
    last_day = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return start, end


def inject_month_css(months_with_samples: set, selected_months: list):
    css = ["<style>"]
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
        background-size: 95% 100%;
        background-position: 4.5% 0;
        padding: 1rem 0;
    }
    """)
    for _, m in MONTHS_IT:
        key = f"m{m:02d}"
        sel = f".st-key-{key} button"

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

        css.append(f"""{sel}:hover {{
            transform: translateY(-2px) scale(1.08);
            border-color: {'#0288D1' if m in months_with_samples else '#616161'} !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
            color: {'#01579B' if m in months_with_samples else '#212121'} !important;
        }}""")

        if m in selected_months:
            css.append(f"""{sel} {{
                background-color: {'#B3E5FC' if m in months_with_samples else 'white'} !important;
                color: #1976D2 !important;
                border: 3px solid #1976D2 !important;
                box-shadow:
                    0 0 0 4px rgba(25,118,210,0.25),
                    0 4px 12px rgba(25,118,210,0.35) !important;
                transform: scale(1.12);
                font-weight: 700 !important;
            }}""")

    css.append("</style>")
    st.markdown("\n".join(css), unsafe_allow_html=True)


def upload_to_minio(df, object_name):
    url = f"s3://datasets/{object_name}.csv"
    df.to_csv(
        url,
        index=False,
        storage_options={
            "key": MINIO_ROOT_USER,
            "secret": MINIO_ROOT_PASSWORD,
            "client_kwargs": {"endpoint_url": MINIO_URL},
        },
    )


def postgres_engine():
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_user = os.getenv("POSTGRES_USER", "user")
    db_password = os.getenv("POSTGRES_PASSWORD", "password")
    db_name = os.getenv("POSTGRES_DB", "spaceflight_db")
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_string)


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


# ---------------- CALLBACKS (NO st.rerun) ----------------
def cb_feast_apply():
    st.session_state.pending_action = "feast_apply"

def cb_generate():
    st.session_state.pending_action = "generate"

def cb_retrain():
    st.session_state.pending_action = "retrain"

def cb_api():
    sm = st.session_state.get("selected_months", [])
    if len(sm) == 1:
        st.session_state.api_call_to_show = int(sm[0])
        st.session_state.selected_months = []

def cb_analyze_start():
    sm = st.session_state.get("selected_months", [])
    if len(sm) == 1:
        st.session_state.analyze_pending = True
        st.session_state.analyze_reference_month = int(sm[0])
        st.session_state.analyze_last_pair = None
        st.session_state.selected_months = []  # step2: scegli current

def cb_analyze_cancel():
    st.session_state.analyze_pending = False
    st.session_state.analyze_reference_month = None
    st.session_state.analyze_last_pair = None


# ---------------- ACTION RUNNER (prima del render UI) ----------------
_init_state()
action = st.session_state.get("pending_action")

# ---- FEAST APPLY ----
if action == "feast_apply":
    st.session_state.pending_action = None
    st.session_state.feast_apply_running = True

    repo_dir = Path(__file__).resolve().parent
    with st.spinner("Eseguo `feast apply`..."):
        try:
            proc = subprocess.run(
                ["feast", "-c", str(repo_dir), "apply"],
                capture_output=True,
                text=True,
            )
            st.session_state.feast_apply_status = "ok" if proc.returncode == 0 else "ko"
            st.session_state.feast_apply_last_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            st.session_state.feast_apply_status = "ko"
            st.session_state.feast_apply_last_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    st.session_state.feast_apply_running = False


# ---- GENERATE ----
elif action == "generate":
    st.session_state.pending_action = None
    sm = st.session_state.get("selected_months", [])
    if len(sm) != 1:
        st.session_state.postgres_error = "Generate richiede ESATTAMENTE 1 mese selezionato."
        st.session_state.generated = False
        st.session_state.selected_months = []
    else:
        month = int(sm[0])
        year = datetime.now(timezone.utc).year
        month_start, month_end = month_bounds_utc(year, month)

        with st.spinner("Genero samples..."):
            try:
                if not os.path.exists("data/reference.csv"):
                    raise FileNotFoundError("File 'reference.csv' not found.")

                num_samples = int(st.session_state.get("num_samples", 10000))

                reference_df = pd.read_csv("data/reference.csv")
                current_df = reference_df.sample(n=num_samples, replace=True).reset_index(drop=True)

                # slider values
                engines_range = st.session_state.get("engines_range", (1.0, 4.0))
                passenger_capacity_range = st.session_state.get("passenger_capacity_range", (2, 10))
                crew_range = st.session_state.get("crew_range", (1.0, 5.0))
                company_rating_range = st.session_state.get("company_rating_range", (3.0, 5.0))
                review_scores_rating_range = st.session_state.get("review_scores_rating_range", (3.5, 35.0))
                price_range = st.session_state.get("price_range", (500.0, 3000.0))

                d_check_prob = st.session_state.get("d_check_prob", 70)
                moon_clearance_prob = st.session_state.get("moon_clearance_prob", 60)
                iata_approved_prob = st.session_state.get("iata_approved_prob", 50)

                generation_timestamp = datetime.now(tz=timezone.utc)

                current_df["engines"] = np.random.randint(int(engines_range[0]), int(engines_range[1]) + 1, len(current_df)).astype(float)
                current_df["passenger_capacity"] = np.random.randint(passenger_capacity_range[0], passenger_capacity_range[1] + 1, len(current_df))
                current_df["crew"] = np.random.uniform(crew_range[0], crew_range[1], len(current_df)).round(1)

                current_df["d_check_complete"] = np.random.random(len(current_df)) < (d_check_prob / 100)
                current_df["moon_clearance_complete"] = np.random.random(len(current_df)) < (moon_clearance_prob / 100)
                current_df["iata_approved"] = np.random.random(len(current_df)) < (iata_approved_prob / 100)

                current_df["company_rating"] = np.random.uniform(company_rating_range[0], company_rating_range[1], len(current_df)).round(1)
                current_df["review_scores_rating"] = np.random.uniform(review_scores_rating_range[0], review_scores_rating_range[1], len(current_df)).round(1)
                current_df["price"] = np.random.uniform(price_range[0], price_range[1], len(current_df)).round(1)

                start_u = int(month_start.timestamp())
                end_u = int(month_end.timestamp())
                rand_u = np.random.randint(start_u, end_u + 1, size=len(current_df), dtype=np.int64)
                current_df["event_timestamp"] = pd.to_datetime(rand_u, unit="s", utc=True)

                st.session_state.generated_start_date = month_start.isoformat()
                st.session_state.generated_end_date = month_end.isoformat()

                st.session_state.generated_data = current_df
                st.session_state.generated = True
                st.session_state.postgres_success = False
                st.session_state.postgres_error = None

                # mark month as generated
                st.session_state.months_with_samples.update([month])
                st.session_state.month_samples_meta[month] = {
                    "last_ts": generation_timestamp.isoformat(),
                    "count": int(num_samples),
                }

                # write to Postgres
                engine = postgres_engine()
                current_df.to_sql("spaceflight_table", engine, if_exists="append", index=False)
                st.session_state.postgres_success = True

                # upload to MinIO
                upload_to_minio(current_df, MINIO_CURRENT_OBJ + generation_timestamp.strftime("%Y-%m-%d_%H-%M-%S"))
                st.session_state.selected_months = []

            except Exception as e:
                st.session_state.generated = False
                st.session_state.postgres_success = False
                st.session_state.postgres_error = str(e)
                st.session_state.selected_months = []


# ---- RETRAIN (submit + polling completo) ----
elif action == "retrain":
    st.session_state.pending_action = None
    st.session_state.retrain_last = None
    st.session_state.retrain_error = None

    sm = st.session_state.get("selected_months", [])
    months_with_samples = st.session_state.get("months_with_samples", set())
    st.session_state.selected_months = []

    if not (len(sm) == 1 and set(sm).issubset(months_with_samples)):
        st.session_state.retrain_error = "Train/retrain richiede ESATTAMENTE 1 mese selezionato e già generato (azzurro)."
    else:
        retrain_url = os.getenv("RETRAIN_URL", "http://training-pipeline:8005/run-pipeline").strip()
        if not retrain_url:
            st.session_state.retrain_error = "RETRAIN_URL non configurata."
        else:
            month = int(sm[0])
            year = datetime.now(timezone.utc).year
            month_start, month_end = month_bounds_utc(year, month)

            payload = {
                "start_date": month_start.isoformat(),
                "end_date": month_end.isoformat(),
                "pipeline": "training",
                "year": year,
                "months": [month],
                "num_samples": int(st.session_state.get("num_samples", 10000)),
            }

            poll_interval_sec = 2
            max_polls = 1800

            with st.spinner("Training/retraining in corso..."):
                try:
                    submit = requests.post(retrain_url, json=payload, timeout=30)
                    if submit.status_code not in (200, 202):
                        st.session_state.retrain_error = f"Submit fallito: {submit.status_code} - {submit.text}"
                    else:
                        submit_json = submit.json()
                        job_id = submit_json.get("job_id")
                        status_path = submit_json.get("status_url")

                        if not job_id:
                            st.session_state.retrain_error = f"Submit OK ma manca job_id: {submit_json}"
                        else:
                            status_url = urljoin(retrain_url, status_path) if status_path else urljoin(retrain_url, f"/run-pipeline/{job_id}")

                            last_status = None
                            for _ in range(max_polls):
                                r = requests.get(status_url, timeout=30)
                                if not r.ok:
                                    st.session_state.retrain_error = f"Errore polling status: {r.status_code} - {r.text}"
                                    break

                                last_status = r.json()
                                state = last_status.get("status")

                                if state in ("completed", "failed"):
                                    break

                                time.sleep(poll_interval_sec)

                            st.session_state.retrain_last = {
                                "job_id": job_id,
                                "status_url": status_url,
                                "last_status": last_status,
                            }
                    st.session_state.selected_months = []

                except Exception as e:
                    st.session_state.retrain_error = f"Errore chiamata training/retrain: {e}"
                    st.session_state.selected_months = []


# ---------------- UI ----------------
st.title("ML Feature Generator")

# --- FEAST APPLY ROW (sinistra + badge vicino) ---

st.markdown("**Create or update a feature store deployment:**")

c_left, c_right = st.columns([1.4, 6.6], vertical_alignment="center")

with c_left:
    st.button(
        "⬆️ Feast apply",
        key="btn_feast_apply",
        disabled=st.session_state.get("feast_apply_running", False),
        on_click=cb_feast_apply,
    )

with c_right:
    status = st.session_state.get("feast_apply_status")

    if status == "ok":
        st.markdown(
            "<span class='feast-badge feast-ok'>✅ Feature successfully loaded on the Feature Store</span>",
            unsafe_allow_html=True
        )
    elif status == "ko":
        st.markdown(
            "<span class='feast-badge feast-ko'>❌ Feature loading failed</span>",
            unsafe_allow_html=True
        )


st.markdown("---")
st.write("Configura features range and generate new samples")

col_samples, _ = st.columns([1, 4])

with col_samples:
    st.number_input(
        "Number of samples",
        min_value=1,
        value=10000,
        step=1,
        key="num_samples"
    )

left_panel, right_panel = st.columns([3.2, 1.3], gap="large")

with left_panel:
    st.subheader("Range delle Features Numeriche")
    ncol1, ncol2 = st.columns(2, gap="medium")
    with ncol1:
        st.slider("Engines - Range", 0.0, 10.0, (1.0, 4.0), 1.0, key="engines_range")
        st.slider("Passenger Capacity - Range", 1, 50, (2, 10), 1, key="passenger_capacity_range")
        st.slider("Crew - Range", 0.0, 20.0, (1.0, 5.0), 1.0, key="crew_range")
    with ncol2:
        st.slider("Company Rating - Range", 0.0, 5.0, (3.0, 5.0), 0.1, key="company_rating_range")
        st.slider("Review Scores Rating - Range", 0.0, 100.0, (3.5, 35.0), 0.1, key="review_scores_rating_range")
        st.slider("Price - Range", 0.0, 10000.0, (500.0, 3000.0), 50.0, key="price_range")

with right_panel:
    st.subheader("Boolean features probability")
    st.slider("D Check Complete (%)", 0, 100, 70, 1, key="d_check_prob")
    st.slider("Moon Clearance Complete (%)", 0, 100, 60, 1, key="moon_clearance_prob")
    st.slider("IATA Approved (%)", 0, 100, 50, 1, key="iata_approved_prob")

st.markdown("---")

months_with_samples = st.session_state.get("months_with_samples", set())
selected_months = st.session_state.get("selected_months", [])

left, right = st.columns([7.2, 2.8], gap="small", vertical_alignment="top")

with left:
    st.subheader("Timeline")
    inject_month_css(months_with_samples, selected_months)

    timeline = st.container(key="timeline_row")
    with timeline:
        mcols = st.columns(12, gap="small")
        for i, (lab, m) in enumerate(MONTHS_IT):
            with mcols[i]:
                st.button(
                    lab,
                    key=f"m{m:02d}",
                    on_click=select_single_month,
                    args=(m,),
                    use_container_width=True,
                )

with right:
    pending = st.session_state.get("analyze_pending", False)
    ref_month = st.session_state.get("analyze_reference_month")

    generate_enabled = (len(selected_months) == 1)
    analyze_enabled = ((not pending) and (len(selected_months) == 1) and set(selected_months).issubset(months_with_samples))
    retrain_enabled = ((len(selected_months) == 1) and set(selected_months).issubset(months_with_samples))
    api_call_enabled = ((len(selected_months) == 1) and set(selected_months).issubset(months_with_samples))

    keypad = st.container(key="action_keypad")
    with keypad:
        r1c1, r1c2 = st.columns(2, gap="xsmall")
        with r1c1:
            st.button("✨ Generate", key="pad_generate", type="primary", disabled=not generate_enabled, on_click=cb_generate)
        with r1c2:
            st.button("🔎 Analyze", key="pad_analyze", disabled=not analyze_enabled, on_click=cb_analyze_start)

        r2c1, r2c2 = st.columns(2, gap="xsmall")
        with r2c1:
            st.button("🔁 Train", key="pad_retrain", disabled=not retrain_enabled, on_click=cb_retrain)
        with r2c2:
            st.button("✖️ Cancel", key="pad_cancel", disabled=not pending, on_click=cb_analyze_cancel)

        r3c1, r3c2 = st.columns(2, gap="xsmall")
        with r3c1:
            st.button("📄 API", key="pad_api", disabled=not api_call_enabled, on_click=cb_api)
        with r3c2:
            st.write("")

    msg = ""
    if pending and ref_month is not None:
        msg = f"Analyze (Step 2): reference = {NUM_TO_LABEL.get(ref_month, ref_month)} → seleziona un mese current già generato (azzurro)."
    else:
        if not generate_enabled:
            msg = "Generate: seleziona ESATTAMENTE 1 mese."
        elif not analyze_enabled:
            msg = "Analyze: seleziona 1 mese già generato (azzurro) per impostarlo come reference."
        elif not retrain_enabled:
            msg = "Train: seleziona 1 mese già generato (azzurro)."
        elif not api_call_enabled:
            msg = "API: seleziona 1 mese già generato (azzurro)."
    st.caption(msg if msg else " ")


# ---------------- API CALL OUTPUT ----------------
api_month = st.session_state.get("api_call_to_show")
if api_month is not None:
    year = datetime.now(timezone.utc).year
    m_start, m_end = month_bounds_utc(year, int(api_month))
    curl_command = f'''curl -X POST http://localhost:3000/batch-scoring \\
  -H "Content-Type: application/json" \\
  -d '{{"start_date": "{m_start.isoformat()}", "end_date": "{m_end.isoformat()}"}}' '''
    st.subheader(f"API call (mese {NUM_TO_LABEL.get(int(api_month), api_month)})")

    api_col, _ = st.columns([2,1])
    with api_col:
        st.code(curl_command, language="bash",
                wrap_lines=True)


# ---------------- ANALYZE STEP 2 (Evidently) ----------------
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
            st.session_state.selected_months = []


# ---------------- POST-GENERATE UI ----------------
if st.session_state.get("generated", False):
    st.markdown("---")
    df_last = st.session_state.generated_data

    if st.session_state.get("postgres_success", False):
        st.success(f"{len(df_last)} samples loaded on the Offline Store")
    elif st.session_state.get("postgres_error"):
        st.error(f"Error during loading on the Offline Store: {st.session_state.postgres_error}")


# ---------------- POST-RETRAIN UI ----------------
if st.session_state.get("retrain_error"):
    st.markdown("---")
    st.error(st.session_state.retrain_error)

if st.session_state.get("retrain_last") is not None:
    st.markdown("---")

    last = st.session_state.retrain_last or {}
    last_status = last.get("last_status") or {}
    state = last_status.get("status")

    if state == "completed":
        st.success("✅ Training completed")
    elif state == "failed":
        st.error("❌ Training failed")
    else:
        # fallback (non dovrebbe capitare, ma safe)
        st.success("✅ Training completed")