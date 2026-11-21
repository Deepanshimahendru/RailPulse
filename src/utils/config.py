"""Central configuration: paths, thresholds and feature switches for RailPulse.

All paths are built with pathlib so they work on Windows, macOS and Linux.
Nothing here reads or writes files.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
UPLOAD_DIR = DATA_DIR / "uploads"
# Operational (current-running) uploads live here, separate from historical data.
OPERATIONAL_DIR = UPLOAD_DIR / "operational_data"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
BACKUP_DIR = BASE_DIR / "backups"

# Historical (training) data
RAW_RAILWAY_FILE = RAW_DIR / "2026-10-10.parquet"
STATIONS_JSON = RAW_DIR / "stations.json"
CLEAN_RAILWAY_FILE = PROCESSED_DIR / "railway_clean.parquet"      # Phase 1 output
FEATURES_FILE = PROCESSED_DIR / "railway_features.parquet"
WEATHER_FILE = PROCESSED_DIR / "weather_hourly.parquet"
STATION_STATE_FILE = PROCESSED_DIR / "station_state_mapping.csv"
HOLIDAY_CALENDAR_FILE = PROCESSED_DIR / "holiday_calendar.csv"

# Model artefacts
MODEL_FILE = MODELS_DIR / "railpulse_model.pkl"
MODEL_V1_FILE = MODELS_DIR / "railpulse_model_v1.pkl"             # backup of the original model
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Reports written by the cleaning step
QUALITY_REPORT_FILE = REPORTS_DIR / "data_quality_report.json"
ROLLOVER_CORRECTIONS_FILE = REPORTS_DIR / "rollover_corrections.csv"

# --------------------------------------------------------------------------
# Data-cleaning thresholds
# --------------------------------------------------------------------------
TIMEZONE = "Asia/Kolkata"

# A scheduled arrival that is later than the same stop's scheduled departure by
# more than this many minutes is treated as a midnight date-rollover error
# (arrival stamped with the departure's date, i.e. one day too late).
ROLLOVER_MIN_GAP_MIN = 720

# Arrival delays below this are far outside any real early arrival and indicate a
# date error. Used only as a validation check after cleaning.
SUSPECT_NEGATIVE_DELAY_MIN = -600

# Very large positive delays are real in Indian Railways data, so they are NOT
# altered; they are only counted in the quality report for review.
HIGH_DELAY_REVIEW_MIN = 600

# --------------------------------------------------------------------------
# Feature switches
# --------------------------------------------------------------------------
# The holiday feature-engineering logic and holiday_calendar.csv are kept in the
# project, but the current calendar is empty (holiday_flag is 0 on every row), so
# the flag carries no information. It stays OFF as a model feature until a
# properly sourced national/regional calendar and enough dated data exist.
USE_HOLIDAY_FEATURE = False
