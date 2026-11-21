import hashlib
import hmac
import html
import io
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import pdfplumber


# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="RailPulse | Railway Intelligence",
    page_icon="🚆",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "railway_features.parquet"
MODEL_PATH = BASE_DIR / "models" / "railpulse_model.pkl"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FEATURES = [
    "category", "station_code", "state", "seq",
    "scheduled_arrival_hour", "scheduled_departure_hour",
    "previous_station_delay", "current_station_delay",
    "rush_hour_flag", "weekend_flag", "holiday_flag",
    "temperature", "humidity", "precipitation", "wind_speed",
    "weather_missing",
]

FAQS = [
    (
        "What is RailPulse?",
        "RailPulse is an academic railway delay prediction and analysis project. "
        "It uses railway journey records and a machine-learning model to estimate "
        "the arrival delay at the next station."
    ),
    (
        "Are the train timings and predictions live?",
        "No. The current demonstration uses historical railway and weather records. "
        "It is not connected to a live railway running-status feed."
    ),
    (
        "How do I check a train?",
        "Open Check Your Train, search by train number or name, choose an available "
        "journey date, and explore its station-wise route and delay estimates."
    ),
    (
        "How does RailPulse estimate the next-station delay?",
        "The model uses information such as the delay already recorded at the current "
        "and previous stations, train and station details, and available weather and "
        "calendar features to estimate the next station's arrival delay."
    ),
    (
        "Does rain always mean a train will be delayed?",
        "No. Weather is only one possible indicator. A weather alert does not prove "
        "that weather caused a delay."
    ),
    (
        "Can anyone upload railway data?",
        "No. The demonstration upload area is available only to accounts configured "
        "by the administrator. Uploaded files are stored separately and do not "
        "automatically change the model or public dataset."
    ),
]


# =========================================================
# DATA AND MODEL
# =========================================================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    data = pd.read_parquet(DATA_PATH)
    if "start_date" in data.columns:
        data["start_date"] = pd.to_datetime(
            data["start_date"], errors="coerce"
        ).dt.date
    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


# =========================================================
# DESIGN SYSTEM — HIGH CONTRAST RAILWAY PORTAL
# =========================================================
st.markdown(
    """
    <style>
    :root {
        --navy: #123451;
        --navy2: #1D4B70;
        --red: #B8322A;
        --ink: #142536;
        --muted: #425466;
        --line: #C8D3DE;
        --paper: #FFFFFF;
        --canvas: #EAF0F5;
        --green: #167344;
        --amber: #9A5B00;
        --blue: #155E9A;
    }

    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif !important;
    }
    .stApp {
        background: var(--canvas) !important;
        color: var(--ink) !important;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--canvas) !important;
    }
    [data-testid="stHeader"] {
        background: rgba(234,240,245,.96) !important;
    }
    .block-container {
        max-width: 1440px !important;
        padding: 1.25rem 2rem 3rem 2rem !important;
    }

    /* Global text contrast */
    .stApp p, .stApp li, .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"] {
        color: var(--ink) !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: var(--navy) !important;
        font-weight: 800 !important;
    }
    [data-testid="stCaptionContainer"] p {
        color: #425466 !important;
    }

    /* Header */
    .rp-header {
        background: #FFFFFF;
        border: 1px solid #C8D3DE;
        border-top: 6px solid #123451;
        border-bottom: 5px solid #B8322A;
        padding: 22px 28px;
        margin-bottom: 18px;
        box-shadow: 0 3px 12px rgba(18,52,81,.07);
    }
    .rp-brand {
        color: #123451 !important;
        font-size: 32px;
        line-height: 1.15;
        font-weight: 900;
        letter-spacing: 1px;
    }
    .rp-subtitle {
        color: #263E53 !important;
        font-size: 14px;
        font-weight: 600;
        margin-top: 5px;
    }
    .rp-kicker {
        color: #B8322A !important;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .rp-section-intro {
        color: #34495D !important;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 14px;
    }

    /* Panels and metric cards */
    .rp-panel {
        background: #FFFFFF;
        border: 1px solid #C8D3DE;
        border-radius: 8px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(18,52,81,.05);
        margin: 10px 0 16px 0;
    }
    .rp-dark-panel {
        background: #123451;
        color: #FFFFFF !important;
        border-left: 6px solid #B8322A;
        border-radius: 8px;
        padding: 24px 26px;
        margin: 22px 0;
    }
    .rp-dark-panel h3, .rp-dark-panel p,
    .rp-dark-panel li, .rp-dark-panel strong {
        color: #FFFFFF !important;
    }
    .rp-dark-panel p, .rp-dark-panel li {
        line-height: 1.65;
    }
    .rp-status {
        background: #FFF8E8;
        border: 1px solid #E6C98E;
        border-left: 5px solid #9A5B00;
        border-radius: 6px;
        color: #573600 !important;
        padding: 13px 17px;
        font-weight: 700;
        margin: 15px 0 20px 0;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #C8D3DE !important;
        border-top: 4px solid #1D4B70 !important;
        border-radius: 8px !important;
        padding: 15px 17px !important;
        box-shadow: 0 2px 8px rgba(18,52,81,.05);
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #34495D !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricValue"] {
        color: #123451 !important;
        font-weight: 900 !important;
    }

    /* Inputs */
    .stTextInput input, .stDateInput input,
    [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: #142536 !important;
        border-color: #8298AA !important;
    }
    input, textarea {
        color: #142536 !important;
    }
    [data-baseweb="select"] * {
        color: #142536 !important;
    }

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button {
        min-height: 44px;
        border-radius: 6px !important;
        border: 1px solid #55738D !important;
        background: #FFFFFF !important;
        color: #123451 !important;
        font-weight: 800 !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #E0EAF2 !important;
        border-color: #123451 !important;
        color: #102D49 !important;
    }
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: #123451 !important;
        color: #FFFFFF !important;
        border: 1px solid #123451 !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover {
        background: #1D4B70 !important;
        color: #FFFFFF !important;
    }

    /* Expanders, tables, uploader */
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #C8D3DE !important;
        border-radius: 7px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #123451 !important;
        font-weight: 800 !important;
    }
    [data-testid="stDataFrame"] {
        background: #FFFFFF !important;
        border: 1px solid #C8D3DE !important;
        border-radius: 7px !important;
    }
    [data-testid="stFileUploader"] {
        background: #FFFFFF !important;
        border: 1px solid #C8D3DE !important;
        border-radius: 7px !important;
        padding: 12px !important;
    }
    [data-testid="stFileUploader"] * {
        color: #123451 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #F8FAFC !important;
        border: 1px dashed #8298AA !important;
    }
    .stAlert p { color: #142536 !important; }
    hr { border-color: #C8D3DE !important; }
    footer { visibility: hidden; }

    /* Metro-style station route */
    .route-wrap {
        background: #FFFFFF;
        border: 1px solid #C8D3DE;
        border-radius: 10px;
        padding: 22px 24px;
        margin: 12px 0 18px 0;
        box-shadow: 0 2px 8px rgba(18,52,81,.05);
    }
    .route-head {
        color: #123451 !important;
        font-size: 19px;
        font-weight: 900;
        margin-bottom: 4px;
    }
    .route-subhead {
        color: #425466 !important;
        font-size: 13px;
        margin-bottom: 18px;
    }
    .station-row {
        display: grid;
        grid-template-columns: 34px minmax(0, 1fr) auto;
        gap: 13px;
        align-items: start;
        min-height: 88px;
    }
    .station-track {
        display: flex;
        flex-direction: column;
        align-items: center;
        height: 100%;
    }
    .station-dot {
        width: 17px;
        height: 17px;
        border-radius: 50%;
        border: 4px solid #1D4B70;
        background: #FFFFFF;
        margin-top: 3px;
        z-index: 2;
    }
    .station-dot.origin { border-color: #167344; background: #E8F5EE; }
    .station-dot.destination { border-color: #B8322A; background: #FCEDEA; }
    .station-line {
        width: 4px;
        min-height: 65px;
        background: #9AAFC1;
        flex: 1;
        margin-top: 3px;
    }
    .station-line.risk { background: #C17A17; }
    .station-name {
        color: #123451 !important;
        font-size: 16px;
        font-weight: 900;
        line-height: 1.3;
    }
    .station-code {
        color: #526779 !important;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .7px;
        margin-left: 7px;
    }
    .station-time {
        color: #263E53 !important;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 4px;
    }
    .station-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 7px;
    }
    .rp-pill {
        display: inline-block;
        border-radius: 20px;
        padding: 4px 9px;
        font-size: 11px;
        font-weight: 900;
        border: 1px solid transparent;
    }
    .rp-green { background: #E8F5EE; color: #145B36 !important; border-color: #9AC9AD; }
    .rp-amber { background: #FFF3D9; color: #704300 !important; border-color: #E7C47C; }
    .rp-red { background: #FCE9E7; color: #8C211B !important; border-color: #E0A09A; }
    .rp-blue { background: #E7F1FA; color: #164D7A !important; border-color: #A7C6E2; }
    .rp-neutral { background: #EDF1F5; color: #34495D !important; border-color: #C8D3DE; }
    .route-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        padding-top: 15px;
        border-top: 1px solid #D8E0E8;
        margin-top: 8px;
    }
    .route-legend span {
        color: #34495D !important;
        font-size: 12px;
        font-weight: 700;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem .8rem 2rem .8rem !important; }
        .rp-brand { font-size: 25px; }
        .route-wrap { padding: 16px 13px; }
        .station-row { grid-template-columns: 27px minmax(0,1fr); }
        .station-row > .station-badges { grid-column: 2; margin-top: -24px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def safe_text(value, fallback="—"):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    return str(value)


def train_label(row):
    number = str(row["train_number"])
    name = (
        str(row["train_name"])
        if "train_name" in row and pd.notna(row["train_name"])
        else ""
    )
    return f"{number} — {name}" if name else number


def get_train_options(data):
    cols = ["train_number"]
    if "train_name" in data.columns:
        cols.append("train_name")
    trains = data[cols].dropna(subset=["train_number"]).drop_duplicates()
    return {
        train_label(row): str(row["train_number"])
        for _, row in trains.iterrows()
    }


def format_time(value):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%I:%M %p").lstrip("0")

    text_value = str(value).strip()
    if not text_value or text_value.lower() in {"nan", "nat", "none"}:
        return "—"

    parsed = pd.to_datetime(text_value, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%I:%M %p").lstrip("0")
    return text_value


def predicted_arrival(scheduled_value, delay_minutes):
    if pd.isna(delay_minutes):
        return "—"
    try:
        parsed = pd.to_datetime(str(scheduled_value), errors="coerce")
        if pd.isna(parsed):
            return "—"
        return (parsed + pd.to_timedelta(float(delay_minutes), unit="m")).strftime(
            "%I:%M %p"
        ).lstrip("0")
    except Exception:
        return "—"


def delay_status(delay):
    if pd.isna(delay):
        return "Estimate unavailable", "rp-neutral"
    if delay <= 5:
        return "Lower delay estimate", "rp-green"
    if delay <= 15:
        return "Possible delay", "rp-amber"
    return "Higher delay estimate", "rp-red"


def weather_status(row):
    missing = row.get("weather_missing", 1)
    precipitation = row.get("precipitation")
    wind = row.get("wind_speed")

    if pd.isna(missing) or missing != 0:
        return None
    tags = []
    if pd.notna(precipitation) and precipitation >= 2:
        tags.append("Rain recorded")
    if pd.notna(wind) and wind >= 30:
        tags.append("High wind recorded")
    return " · ".join(tags) if tags else None


def station_name(row):
    for col in ["station_name", "station", "station_code"]:
        value = row.get(col)
        if value is not None and pd.notna(value):
            return str(value)
    return "Unknown station"


def make_prediction_results(journey):
    """Predict next-station delay from each current-station feature row."""
    missing = [col for col in MODEL_FEATURES if col not in journey.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    if model is None:
        raise ValueError("The trained model was not found.")

    ordered = journey.sort_values("seq").reset_index(drop=True).copy()
    predictions = model.predict(ordered[MODEL_FEATURES])
    ordered["next_station_code"] = ordered["station_code"].shift(-1)
    ordered["predicted_next_delay_min"] = predictions
    ordered["next_scheduled_arrival"] = ordered["scheduled_arrival"].shift(-1) if "scheduled_arrival" in ordered.columns else np.nan
    ordered["next_actual_arrival"] = ordered["actual_arrival"].shift(-1) if "actual_arrival" in ordered.columns else np.nan
    ordered["next_actual_delay_min"] = ordered["arrival_delay_min"].shift(-1) if "arrival_delay_min" in ordered.columns else np.nan
    return ordered.dropna(subset=["next_station_code"]).copy()


def render_route_timeline(journey, prediction_results):
    """Render a metro-style route. Historical records are labelled honestly."""
    route = journey.sort_values("seq").reset_index(drop=True)
    prediction_by_current = {}
    for _, row in prediction_results.iterrows():
        prediction_by_current[str(row["station_code"])] = row

    st.markdown(
        """
        <div class="route-wrap">
          <div class="route-head">Station-wise journey route</div>
          <div class="route-subhead">
            Metro-style route view · Historical journey replay with model estimates
          </div>
        """,
        unsafe_allow_html=True,
    )

    for index, row in route.iterrows():
        code = str(row.get("station_code", ""))
        name = html.escape(station_name(row))
        safe_code = html.escape(code)
        scheduled = format_time(row.get("scheduled_arrival"))
        actual = format_time(row.get("actual_arrival"))
        actual_delay = row.get("arrival_delay_min")
        weather = weather_status(row)

        is_origin = index == 0
        is_destination = index == len(route) - 1
        dot_class = "origin" if is_origin else ("destination" if is_destination else "")

        pills = []
        if is_origin:
            pills.append('<span class="rp-pill rp-green">Origin</span>')
        elif is_destination:
            pills.append('<span class="rp-pill rp-red">Destination</span>')

        if pd.notna(actual_delay):
            actual_label = f"Recorded delay: {float(actual_delay):.0f} min"
            actual_class = "rp-green" if actual_delay <= 5 else ("rp-amber" if actual_delay <= 15 else "rp-red")
            pills.append(f'<span class="rp-pill {actual_class}">{actual_label}</span>')

        if weather:
            pills.append(f'<span class="rp-pill rp-blue">☔ {html.escape(weather)}</span>')

        # Prediction for this station comes from the previous station's row.
        predicted_delay = np.nan
        predicted_time = "—"
        if index > 0:
            previous_code = str(route.iloc[index - 1].get("station_code", ""))
            previous_prediction = prediction_by_current.get(previous_code)
            if previous_prediction is not None:
                predicted_delay = previous_prediction.get("predicted_next_delay_min")
                predicted_time = predicted_arrival(
                    row.get("scheduled_arrival"), predicted_delay
                )

        if index > 0 and pd.notna(predicted_delay):
            label, css_class = delay_status(predicted_delay)
            pills.append(
                f'<span class="rp-pill {css_class}">Model estimate: {float(predicted_delay):.0f} min</span>'
            )
            pills.append(
                f'<span class="rp-pill rp-neutral">{html.escape(label)}</span>'
            )

        connector = ""
        if not is_destination:
            next_code = str(route.iloc[index + 1].get("station_code", ""))
            connector_class = "risk"
            if code in prediction_by_current:
                d = prediction_by_current[code].get("predicted_next_delay_min")
                if pd.notna(d) and d <= 5:
                    connector_class = ""
            connector = f'<div class="station-line {connector_class}"></div>'

        if index == 0:
            timing_line = f"Scheduled arrival: <b>{html.escape(scheduled)}</b> &nbsp; · &nbsp; Recorded actual arrival: <b>{html.escape(actual)}</b>"
        else:
            timing_line = (
                f"Scheduled arrival: <b>{html.escape(scheduled)}</b> &nbsp; · &nbsp; "
                f"Recorded actual arrival: <b>{html.escape(actual)}</b><br>"
                f"Model-estimated arrival (from previous station): <b>{html.escape(predicted_time)}</b>"
            )

        st.markdown(
            f"""
            <div class="station-row">
              <div class="station-track">
                <div class="station-dot {dot_class}"></div>
                {connector}
              </div>
              <div>
                <div class="station-name">{name}<span class="station-code">{safe_code}</span></div>
                <div class="station-time">{timing_line}</div>
                <div class="station-pills">{''.join(pills)}</div>
              </div>
              <div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
          <div class="route-legend">
            <span>🟢 Lower estimated delay</span>
            <span>🟠 Possible delay</span>
            <span>🔴 Higher estimated delay</span>
            <span>🔵 Weather observation</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Important: This is a historical-data demonstration, not a live journey tracker. "
        "Recorded actual timings belong to the selected historical journey. Model estimates "
        "are generated from the preceding station's feature row. Risk colours are simple "
        "delay bands, not calibrated probabilities."
    )


def explain_delay_factors(row):
    indicators = []

    previous_delay = row.get("previous_station_delay")
    if pd.notna(previous_delay) and previous_delay > 5:
        indicators.append(
            f"The train was already about {previous_delay:.0f} minutes late "
            "at an earlier station."
        )

    current_delay = row.get("current_station_delay")
    if pd.notna(current_delay) and current_delay > 5:
        indicators.append(
            f"The train was about {current_delay:.0f} minutes late at the current station."
        )

    if row.get("rush_hour_flag") == 1:
        indicators.append("The journey record falls in a flagged busy travel period.")

    if row.get("holiday_flag") == 1:
        indicators.append("The available calendar marks this date as a holiday.")

    if row.get("weather_missing") == 0:
        precipitation = row.get("precipitation")
        wind = row.get("wind_speed")
        if pd.notna(precipitation) and precipitation >= 2:
            indicators.append("Recorded precipitation is elevated for this observation.")
        if pd.notna(wind) and wind >= 30:
            indicators.append("Recorded wind speed is elevated for this observation.")

    if indicators:
        for item in indicators:
            st.markdown(f"- {item}")
    else:
        st.info(
            "No strong indicators are available in this record. "
            "RailPulse does not invent a reason when supporting data is missing."
        )

    st.caption("These are data indicators, not confirmed causes of a delay.")


def employee_accounts():
    try:
        return dict(st.secrets.get("employee_accounts", {}))
    except Exception:
        return {}


def password_hash(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def authenticate_employee(username, password):
    stored_hash = employee_accounts().get(username)
    if not stored_hash:
        return False
    return hmac.compare_digest(str(stored_hash), password_hash(password))


# =========================================================
# HEADER AND NAVIGATION
# =========================================================
st.markdown(
    """
    <div class="rp-header">
      <div class="rp-kicker">Railway intelligence & delay analysis</div>
      <div class="rp-brand">🚆 RAILPULSE</div>
      <div class="rp-subtitle">Railway Delay Prediction and Analysis System</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "selected_train_number" not in st.session_state:
    st.session_state.selected_train_number = None
if "selected_journey_date" not in st.session_state:
    st.session_state.selected_journey_date = None
if "show_delay_analysis" not in st.session_state:
    st.session_state.show_delay_analysis = False

pages = ["Home", "Check Your Train", "Employee Login"]
nav_cols = st.columns(3)
for i, page_name in enumerate(pages):
    with nav_cols[i]:
        if st.button(
            page_name,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if st.session_state.page == page_name else "secondary",
        ):
            st.session_state.page = page_name
            st.rerun()

st.divider()

if df is None:
    st.error(
        "Railway dataset not found at data/processed/railway_features.parquet. "
        "Check the project data folder."
    )
    st.stop()

required_columns = {"train_number", "station_code", "seq"}
if not required_columns.issubset(df.columns):
    st.error(f"Dataset is missing required columns: {sorted(required_columns - set(df.columns))}")
    st.stop()


# =========================================================
# HOME PAGE
# =========================================================
if st.session_state.page == "Home":
    st.markdown('<div class="rp-kicker">System overview</div>', unsafe_allow_html=True)
    st.markdown("# Railway Network Overview")
    st.markdown(
        '<div class="rp-section-intro">A snapshot of the historical railway records '
        'currently available in the RailPulse demonstration.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trains in dataset", f"{df['train_number'].nunique():,}")
    c2.metric("Station codes", f"{df['station_code'].nunique():,}")

    if "start_date" in df.columns and df["start_date"].notna().any():
        min_date = min(df["start_date"].dropna())
        max_date = max(df["start_date"].dropna())
        c3.metric("Data coverage", f"{min_date:%d %b} – {max_date:%d %b %Y}")
    else:
        c3.metric("Data coverage", "Not available")

    c4.metric("ML model", "Available" if model is not None else "Not found")

    st.markdown(
        """
        <div class="rp-status">
          DEMONSTRATION MODE · This version uses historical railway and weather records.
          It is not connected to a live train-running feed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="rp-panel">
          <div class="rp-kicker">Passenger services</div>
          <h3 style="color:#123451;margin:0 0 8px 0;">Check your train journey</h3>
          <p style="color:#263E53;margin:0;">
            Search for a train, select an available journey date, follow its station-wise
            route and explore model-estimated next-station delays.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("CHECK YOUR TRAIN  →", type="primary", use_container_width=True):
        st.session_state.page = "Check Your Train"
        st.rerun()

    st.markdown(
        """
        <div class="rp-dark-panel">
          <h3>About RailPulse</h3>
          <p>
            RailPulse is an independent academic project that studies railway running
            records and uses machine learning to estimate arrival delays at upcoming
            stations. The aim is to present journey information and delay estimates in
            a clear, passenger-friendly format.
          </p>
          <ul>
            <li>Train-wise journey and station information</li>
            <li>Metro-style station route timeline</li>
            <li>Next-station delay estimates with supporting data indicators</li>
            <li>Restricted employee data-upload demonstration</li>
          </ul>
          <p>
            RailPulse is not an official Indian Railways or CRIS service.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Frequently Asked Questions")
    st.caption("Select a question to expand its answer.")
    for question, answer in FAQS:
        with st.expander(question, expanded=False):
            st.write(answer)


# =========================================================
# CHECK YOUR TRAIN
# =========================================================
elif st.session_state.page == "Check Your Train":
    st.markdown('<div class="rp-kicker">Passenger service</div>', unsafe_allow_html=True)
    st.markdown("# Check Your Train")
    st.markdown(
        '<div class="rp-section-intro">Search for a train and select a journey date '
        'available in the historical dataset.</div>',
        unsafe_allow_html=True,
    )

    train_options = get_train_options(df)
    labels = sorted(train_options.keys())

    search = st.text_input(
        "Search by train number or name",
        placeholder="For example: enter a train number or train name",
        key="train_search_input",
    )
    filtered_labels = (
        [label for label in labels if search.strip().lower() in label.lower()]
        if search.strip()
        else labels
    )

    if not filtered_labels:
        st.warning("No matching train was found in the available records.")
        st.stop()

    default_index = 0
    if st.session_state.selected_train_number is not None:
        for index, label in enumerate(filtered_labels):
            if train_options[label] == st.session_state.selected_train_number:
                default_index = index
                break

    selected_label = st.selectbox(
        "Select train",
        filtered_labels,
        index=default_index,
        key="selected_train_label",
    )
    selected_train = train_options[selected_label]
    st.session_state.selected_train_number = selected_train

    train_data = df[df["train_number"].astype(str) == selected_train].copy()
    available_dates = (
        sorted(train_data["start_date"].dropna().unique())
        if "start_date" in train_data.columns
        else []
    )
    if not available_dates:
        st.warning("No journey dates are available for this train.")
        st.stop()

    date_index = 0
    if st.session_state.selected_journey_date in available_dates:
        date_index = available_dates.index(st.session_state.selected_journey_date)

    selected_date = st.selectbox(
        "Select journey date",
        available_dates,
        index=date_index,
        format_func=lambda value: value.strftime("%d %B %Y"),
        key="selected_train_date",
    )
    st.session_state.selected_journey_date = selected_date

    journey = (
        train_data[train_data["start_date"] == selected_date]
        .sort_values("seq")
        .reset_index(drop=True)
    )

    train_name_value = (
        str(journey["train_name"].dropna().iloc[0])
        if "train_name" in journey.columns and journey["train_name"].notna().any()
        else selected_train
    )

    st.markdown(
        f"""
        <div class="rp-panel">
          <div class="rp-kicker">Selected journey</div>
          <h3 style="color:#123451;margin:0;">{html.escape(train_name_value)}</h3>
          <p style="color:#34495D;margin:5px 0 0 0;">
            Train No. {html.escape(selected_train)} &nbsp; | &nbsp;
            {selected_date:%d %B %Y} &nbsp; | &nbsp;
            {len(journey)} station records
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Route timeline")
    st.caption(
        "Follow the stations in journey order. The geographical map has been removed; "
        "this route diagram focuses on station sequence and timings."
    )

    prediction_results = None
    prediction_error = None
    if model is not None and len(journey) >= 2:
        try:
            prediction_results = make_prediction_results(journey)
        except Exception as exc:
            prediction_error = str(exc)

    if prediction_results is not None:
        render_route_timeline(journey, prediction_results)
    else:
        st.warning(
            "The route can be displayed, but next-station model estimates are unavailable "
            "for this selection."
        )
        if prediction_error:
            st.caption(f"Prediction detail: {prediction_error}")
        # Basic route fallback, without pretending there are model predictions.
        fallback = journey.sort_values("seq")
        st.markdown('<div class="route-wrap">', unsafe_allow_html=True)
        for index, (_, row) in enumerate(fallback.iterrows()):
            station = html.escape(station_name(row))
            code = html.escape(str(row.get("station_code", "")))
            scheduled = html.escape(format_time(row.get("scheduled_arrival")))
            actual = html.escape(format_time(row.get("actual_arrival")))
            st.markdown(
                f"**{index + 1}. {station} ({code})**  \n"
                f"Scheduled: {scheduled} · Recorded actual: {actual}"
            )
            if index != len(fallback) - 1:
                st.markdown("│")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    if st.button("VIEW DETAILED DELAY ANALYSIS", type="primary", use_container_width=True):
        st.session_state.show_delay_analysis = not st.session_state.show_delay_analysis

    if st.session_state.show_delay_analysis:
        st.markdown("## Detailed Delay Analysis")
        if model is None:
            st.error("The trained prediction model was not found.")
        elif prediction_results is None:
            st.error(prediction_error or "Predictions could not be generated.")
        else:
            results = prediction_results.copy()
            results["predicted_next_delay_min"] = pd.to_numeric(
                results["predicted_next_delay_min"], errors="coerce"
            )

            valid_predictions = results["predicted_next_delay_min"].dropna()
            if valid_predictions.empty:
                st.warning("The model did not return usable predictions for this journey.")
            else:
                average_delay = valid_predictions.mean()
                highest_delay = valid_predictions.max()
                high_risk_count = int((valid_predictions > 15).sum())

                m1, m2, m3 = st.columns(3)
                m1.metric("Stations with estimates", len(valid_predictions))
                m2.metric("Average estimated delay", f"{average_delay:.1f} min")
                m3.metric("Stations above 15 min estimate", high_risk_count)

                st.info(
                    "These are historical-data model estimates for the selected journey. "
                    "They are not live running predictions. Negative delay values can "
                    "represent an estimated early arrival."
                )

                st.markdown("### Estimated delay at the next station")
                chart_data = results[
                    ["next_station_code", "predicted_next_delay_min"]
                ].dropna().copy()
                chart_data["predicted_next_delay_min"] = chart_data[
                    "predicted_next_delay_min"
                ].round(1)

                fig = px.line(
                    chart_data,
                    x="next_station_code",
                    y="predicted_next_delay_min",
                    markers=True,
                    labels={
                        "next_station_code": "Next station",
                        "predicted_next_delay_min": "Estimated arrival delay (minutes)",
                    },
                )
                fig.update_traces(
                    line=dict(color="#174A70", width=3),
                    marker=dict(size=9, color="#B8322A"),
                    hovertemplate="Station: %{x}<br>Estimated delay: %{y:.1f} min<extra></extra>",
                )
                fig.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#142536", size=13),
                    xaxis=dict(
                        title="Next station",
                        showgrid=False,
                        tickangle=-35,
                        tickfont=dict(color="#263E53"),
                    ),
                    yaxis=dict(
                        title="Estimated delay (minutes)",
                        showgrid=True,
                        gridcolor="#DCE4EB",
                        zeroline=True,
                        zerolinecolor="#8298AA",
                        tickfont=dict(color="#263E53"),
                    ),
                    margin=dict(l=15, r=15, t=25, b=80),
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### What information may influence the estimate?")
                explain_row = results.iloc[0]
                explain_delay_factors(explain_row)

                st.markdown("### Station-wise prediction details")
                display = results[
                    [
                        col for col in [
                            "station_code",
                            "next_station_code",
                            "current_station_delay",
                            "previous_station_delay",
                            "predicted_next_delay_min",
                            "next_scheduled_arrival",
                            "rush_hour_flag",
                            "holiday_flag",
                            "precipitation",
                            "wind_speed",
                        ]
                        if col in results.columns
                    ]
                ].copy()
                if "predicted_next_delay_min" in display.columns:
                    display["predicted_next_delay_min"] = display[
                        "predicted_next_delay_min"
                    ].round(1)
                display = display.rename(
                    columns={
                        "station_code": "Current station",
                        "next_station_code": "Next station",
                        "current_station_delay": "Current delay (min)",
                        "previous_station_delay": "Previous delay (min)",
                        "predicted_next_delay_min": "Predicted next delay (min)",
                        "next_scheduled_arrival": "Next scheduled arrival",
                        "rush_hour_flag": "Rush hour flag",
                        "holiday_flag": "Holiday flag",
                        "precipitation": "Precipitation",
                        "wind_speed": "Wind speed",
                    }
                )
                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                )


# =========================================================
# EMPLOYEE LOGIN AND UPLOAD
# =========================================================
elif st.session_state.page == "Employee Login":
    st.markdown('<div class="rp-kicker">Restricted area</div>', unsafe_allow_html=True)
    st.markdown("# Employee Login")
    st.markdown(
        '<div class="rp-section-intro">Access is limited to accounts configured '
        'by the RailPulse administrator.</div>',
        unsafe_allow_html=True,
    )

    if "employee_authenticated" not in st.session_state:
        st.session_state.employee_authenticated = False
    if "employee_username" not in st.session_state:
        st.session_state.employee_username = ""

    if not st.session_state.employee_authenticated:
        accounts = employee_accounts()
        if not accounts:
            st.warning(
                "No employee accounts are configured yet. Add approved accounts "
                "in Streamlit secrets before enabling uploads."
            )
            st.code(
                '[employee_accounts]\n'
                'employee1 = "PASTE_SHA256_PASSWORD_HASH_HERE"',
                language="toml",
            )
            st.caption(
                "Generate a demo password hash locally with: "
                "python -c \"import hashlib; print(hashlib.sha256('YOUR_PASSWORD'.encode()).hexdigest())\""
            )

        with st.form("employee_login_form"):
            username = st.text_input("Employee username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "SIGN IN", type="primary", use_container_width=True
            )

        if submitted:
            if authenticate_employee(username.strip(), password):
                st.session_state.employee_authenticated = True
                st.session_state.employee_username = username.strip()
                st.rerun()
            else:
                st.error("Login failed. Check your credentials or contact the administrator.")

    else:
        left, right = st.columns([3, 1])
        with left:
            st.success(f"Signed in as {st.session_state.employee_username}")
        with right:
            if st.button("SIGN OUT", use_container_width=True):
                st.session_state.employee_authenticated = False
                st.session_state.employee_username = ""
                st.rerun()

        st.markdown("## Railway Dataset Upload")
        st.write(
            "Upload a file for preview and basic validation. Files are stored separately; "
            "they do not replace the processed dataset or automatically retrain the model."
        )

        uploaded_file = st.file_uploader(
            "Choose railway data file",
            type=["csv", "xlsx", "xls", "pdf"],
            key="employee_upload",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            upload_df = None
            extracted_tables = []

            try:
                suffix = Path(uploaded_file.name).suffix.lower()
                if suffix == ".csv":
                    upload_df = pd.read_csv(io.BytesIO(file_bytes))
                elif suffix in [".xlsx", ".xls"]:
                    upload_df = pd.read_excel(io.BytesIO(file_bytes))
                elif suffix == ".pdf":
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        for page_number, pdf_page in enumerate(pdf.pages, start=1):
                            for table in pdf_page.extract_tables():
                                if table:
                                    extracted_tables.append(
                                        (page_number, pd.DataFrame(table))
                                    )

                st.markdown("### Upload preview")
                if upload_df is not None:
                    a, b, c = st.columns(3)
                    a.metric("Records", f"{len(upload_df):,}")
                    b.metric("Columns", len(upload_df.columns))
                    c.metric("Missing values", f"{int(upload_df.isna().sum().sum()):,}")
                    st.dataframe(
                        upload_df.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

                    required = ["train_number", "station_code", "seq"]
                    missing = [col for col in required if col not in upload_df.columns]
                    if missing:
                        st.error(f"Validation failed: missing columns {missing}")
                    else:
                        st.success("Basic validation passed: required columns are present.")

                elif extracted_tables:
                    for table_number, (page_number, table) in enumerate(
                        extracted_tables, start=1
                    ):
                        st.caption(f"Page {page_number} · Table {table_number}")
                        st.dataframe(
                            table,
                            use_container_width=True,
                            hide_index=True,
                        )
                    st.info(
                        "PDF tables were extracted for review. Confirm their structure "
                        "before converting them into operational records."
                    )
                else:
                    st.warning(
                        "No tables were detected in this PDF. Scanned PDFs may require OCR."
                    )

                if st.button("SAVE UPLOADED FILE SEPARATELY", type="primary"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
                    saved_path = UPLOAD_DIR / f"{timestamp}_{safe_name}"
                    saved_path.write_bytes(file_bytes)
                    st.success(f"File saved to data/uploads/{saved_path.name}")

                st.caption(
                    "Production deployment should add stronger password storage, "
                    "identity verification, role permissions, malware scanning and "
                    "a tamper-resistant audit log."
                )

            except Exception as exc:
                st.error(f"File processing error: {exc}")


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "RailPulse · Independent academic railway delay prediction demonstration · "
)
