"""Phase 1 - data cleaning for the raw railway running records.

Run from the project root:

    python -m src.data_preprocessing

What it does (no rows are ever dropped, and the raw file is never modified):

1. Standardises types: train numbers stay text (leading zeros kept), timestamps are
   timezone-aware IST.
2. Fixes the midnight date-rollover error: a scheduled arrival at ~23:xx that carries
   the *departure's* date (one day too late) is moved back one day. A row is only
   corrected when the corrected time also fits the journey order
   (previous departure <= arrival <= own departure); otherwise it is reported as
   unresolved and left unchanged.
3. Recomputes arrival/departure delays as actual - scheduled. Missing actual or
   scheduled times stay missing - nothing is imputed.
4. Adds audit/structure flags (origin, destination, unscheduled stop, repeated
   station, observed arrival/departure, route position) so later steps can build
   valid consecutive-station targets.
5. Validates the result and writes a data-quality report.

Outputs (existing outputs are backed up first):
    data/processed/railway_clean.parquet
    reports/data_quality_report.json
    reports/rollover_corrections.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import config
from src.utils.helpers import backup_file, ensure_dir, to_jsonable

JOURNEY_KEY = ["train_number", "start_date"]
TIME_COLUMNS = [
    "scheduled_arrival",
    "scheduled_departure",
    "actual_arrival",
    "actual_departure",
]
REQUIRED_COLUMNS = JOURNEY_KEY + [
    "seq",
    "station_code",
    "status",
    "arrival_delay_min",
    "departure_delay_min",
] + TIME_COLUMNS


class DataValidationError(RuntimeError):
    """Raised when the cleaned data fails a validation check."""


# --------------------------------------------------------------------------
# Step 1 - standardise
# --------------------------------------------------------------------------
def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise types and sort rows by journey and stop sequence."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")

    out = df.copy()
    out["train_number"] = out["train_number"].astype(str).str.strip()
    out["station_code"] = out["station_code"].astype(str).str.strip()
    out["start_date"] = pd.to_datetime(out["start_date"]).dt.date
    out["seq"] = out["seq"].astype(int)

    for column in TIME_COLUMNS:
        series = pd.to_datetime(out[column], errors="coerce")
        if series.dt.tz is None:
            series = series.dt.tz_localize(config.TIMEZONE)
        else:
            series = series.dt.tz_convert(config.TIMEZONE)
        out[column] = series

    return out.sort_values(JOURNEY_KEY + ["seq"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# Step 2 - midnight rollover
# --------------------------------------------------------------------------
def fix_midnight_rollover(df: pd.DataFrame):
    """Correct arrival timestamps that carry the departure's (next) date.

    Returns (clean_df, corrections_df, unresolved_df). ``df`` must be standardised.
    """
    out = df.copy()
    grouped = out.groupby(JOURNEY_KEY, sort=False)
    previous_departure = grouped["scheduled_departure"].shift(1)

    gap_minutes = (
        out["scheduled_arrival"] - out["scheduled_departure"]
    ).dt.total_seconds() / 60.0
    candidate = gap_minutes > config.ROLLOVER_MIN_GAP_MIN

    shifted = out["scheduled_arrival"] - pd.Timedelta(days=1)
    fits_after_previous = previous_departure.isna() | (shifted >= previous_departure)
    fits_own_departure = shifted <= out["scheduled_departure"]
    fixable = (candidate & fits_after_previous & fits_own_departure).fillna(False)

    still_inverted = (out["scheduled_arrival"] > out["scheduled_departure"]).fillna(False)
    unresolved_mask = still_inverted & ~fixable

    before = out.loc[fixable, ["scheduled_arrival", "arrival_delay_min"]].copy()
    out.loc[fixable, "scheduled_arrival"] = shifted[fixable]
    out["rollover_corrected"] = fixable.astype(bool)

    corrections = out.loc[
        fixable, JOURNEY_KEY + ["seq", "station_code", "scheduled_arrival", "scheduled_departure", "actual_arrival"]
    ].copy()
    corrections = corrections.rename(columns={"scheduled_arrival": "scheduled_arrival_corrected"})
    corrections.insert(
        corrections.columns.get_loc("scheduled_arrival_corrected"),
        "scheduled_arrival_original",
        before["scheduled_arrival"],
    )
    corrections["arrival_delay_min_original"] = before["arrival_delay_min"]
    corrections["arrival_delay_min_corrected"] = (
        corrections["actual_arrival"] - corrections["scheduled_arrival_corrected"]
    ).dt.total_seconds() / 60.0

    unresolved = out.loc[
        unresolved_mask,
        JOURNEY_KEY + ["seq", "station_code", "scheduled_arrival", "scheduled_departure"],
    ].copy()
    return out, corrections.reset_index(drop=True), unresolved.reset_index(drop=True)


# --------------------------------------------------------------------------
# Step 3 - delays
# --------------------------------------------------------------------------
def recompute_delays(df: pd.DataFrame) -> pd.DataFrame:
    """delay = actual - scheduled (minutes). Stays NaN when either time is missing."""
    out = df.copy()
    out["arrival_delay_min"] = (
        out["actual_arrival"] - out["scheduled_arrival"]
    ).dt.total_seconds() / 60.0
    out["departure_delay_min"] = (
        out["actual_departure"] - out["scheduled_departure"]
    ).dt.total_seconds() / 60.0
    return out


# --------------------------------------------------------------------------
# Step 4 - structure / audit flags
# --------------------------------------------------------------------------
def add_structure_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add flags that describe each stop's role and whether it was observed."""
    out = df.copy()

    out["is_unscheduled_stop"] = (
        out["scheduled_arrival"].isna() & out["scheduled_departure"].isna()
    )
    scheduled = ~out["is_unscheduled_stop"]

    # Position among scheduled stops only (1..n). Unscheduled extra stops get <NA>.
    route_pos = pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"), index=out.index)
    route_pos.loc[scheduled] = (
        out.loc[scheduled].groupby(JOURNEY_KEY, sort=False).cumcount() + 1
    ).astype("Int64")
    out["route_pos"] = route_pos

    last_pos = out.groupby(JOURNEY_KEY, sort=False)["route_pos"].transform("max")
    out["is_origin"] = (out["route_pos"] == 1).fillna(False).astype(bool)
    out["is_destination"] = (out["route_pos"] == last_pos).fillna(False).astype(bool)

    station_group = out.groupby(JOURNEY_KEY + ["station_code"], sort=False)
    out["station_visit_no"] = station_group.cumcount() + 1
    out["station_repeated_in_journey"] = station_group["seq"].transform("size") > 1

    out["arrival_observed"] = out["arrival_delay_min"].notna()
    out["departure_observed"] = out["departure_delay_min"].notna()
    return out


# --------------------------------------------------------------------------
# Step 5 - validation
# --------------------------------------------------------------------------
def validate_clean(clean: pd.DataFrame, raw_rows: int | None = None) -> list[dict]:
    """Return a list of {name, passed, detail} checks. Does not raise."""
    checks: list[dict] = []

    def add(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    if raw_rows is not None:
        add("row_count_preserved", len(clean) == raw_rows, f"raw={raw_rows}, clean={len(clean)}")

    dup = clean.duplicated(JOURNEY_KEY + ["seq"]).sum()
    add("unique_train_date_seq", dup == 0, f"{dup} duplicate keys")

    ordered = clean.sort_values(JOURNEY_KEY + ["seq"], kind="mergesort").index.equals(clean.index)
    add("sorted_by_journey_and_seq", ordered)

    inverted = (clean["scheduled_arrival"] > clean["scheduled_departure"]).sum()
    add("scheduled_arrival_not_after_departure", inverted == 0, f"{inverted} rows")

    grouped = clean.groupby(JOURNEY_KEY, sort=False)
    prev_dep = grouped["scheduled_departure"].shift(1)
    early = (clean["scheduled_arrival"] < prev_dep).sum()
    add("arrival_not_before_previous_departure", early == 0, f"{early} rows")
    backwards = (clean["scheduled_departure"] < prev_dep).sum()
    add("departure_not_before_previous_departure", backwards == 0, f"{backwards} rows")

    recomputed = (clean["actual_arrival"] - clean["scheduled_arrival"]).dt.total_seconds() / 60.0
    arr_mismatch = ((recomputed - clean["arrival_delay_min"]).abs() > 0.01).sum()
    both_nan_ok = (recomputed.isna() == clean["arrival_delay_min"].isna()).all()
    add("arrival_delay_matches_times", arr_mismatch == 0 and both_nan_ok, f"{arr_mismatch} mismatches")

    suspect = (clean["arrival_delay_min"] < config.SUSPECT_NEGATIVE_DELAY_MIN).sum()
    add(
        "no_rollover_sized_negative_delays",
        suspect == 0,
        f"{suspect} rows below {config.SUSPECT_NEGATIVE_DELAY_MIN} min",
    )

    tz_ok = all(str(clean[c].dt.tz) == config.TIMEZONE for c in TIME_COLUMNS)
    add("timestamps_are_ist", tz_ok)
    return checks


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def clean_railway_data(raw: pd.DataFrame):
    """Run all cleaning steps. Returns (clean_df, corrections_df, unresolved_df)."""
    standardised = standardise(raw)
    fixed, corrections, unresolved = fix_midnight_rollover(standardised)
    with_delays = recompute_delays(fixed)
    clean = add_structure_flags(with_delays)
    return clean, corrections, unresolved


def build_quality_report(raw: pd.DataFrame, clean: pd.DataFrame, corrections, unresolved, checks) -> dict:
    raw_std = standardise(raw)
    key = JOURNEY_KEY

    # How many stored delay values changed (should equal corrected rows with an actual time).
    old = raw_std["arrival_delay_min"]
    new = clean["arrival_delay_min"]
    changed = int((((old - new).abs() > 0.5) | (old.isna() != new.isna())).sum())

    scheduled_stops = clean[clean["scheduled_arrival"].notna()]
    by_status = {}
    for status, part in clean.groupby("status"):
        sched = part[part["scheduled_arrival"].notna()]
        by_status[str(status)] = {
            "rows": int(len(part)),
            "journeys": int(part.groupby(key).ngroups),
            "stops_with_scheduled_arrival": int(len(sched)),
            "arrival_observed_share": (
                round(float(sched["arrival_observed"].mean()), 4) if len(sched) else None
            ),
        }

    journey_obs = (
        scheduled_stops.groupby(key)["arrival_observed"].agg(["all", "sum", "size"])
    )
    status_by_journey = clean.groupby(key)["status"].first()
    completed_idx = status_by_journey[status_by_journey == "completed"].index
    completed = journey_obs.loc[journey_obs.index.intersection(completed_idx)]

    rows_by_date = clean.groupby("start_date").size()
    earlier = rows_by_date.iloc[:-1]
    partial_last_day = bool(
        len(earlier) > 0 and rows_by_date.iloc[-1] < 0.5 * float(earlier.median())
    )

    repeated = clean[clean["station_repeated_in_journey"]]
    seq_span = clean.groupby(key)["seq"].agg(["min", "max", "size"])
    non_contiguous = int(((seq_span["max"] - seq_span["min"] + 1) != seq_span["size"]).sum())

    delay = clean["arrival_delay_min"]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "rows": int(len(raw)),
            "journeys": int(clean.groupby(key).ngroups),
            "trains": int(clean["train_number"].nunique()),
            "stations": int(clean["station_code"].nunique()),
            "start_date_min": clean["start_date"].min(),
            "start_date_max": clean["start_date"].max(),
            "latest_scheduled_time": clean[["scheduled_arrival", "scheduled_departure"]].max().max(),
            "latest_actual_time": clean[["actual_arrival", "actual_departure"]].max().max(),
        },
        "midnight_rollover": {
            "rule": (
                "scheduled_arrival later than same-stop scheduled_departure by > "
                f"{config.ROLLOVER_MIN_GAP_MIN} min and the arrival minus one day still fits "
                "previous departure <= arrival <= own departure"
            ),
            "corrected_rows": int(len(corrections)),
            "unresolved_rows": int(len(unresolved)),
            "delay_values_changed": changed,
            "corrected_delay_range_min": (
                [float(corrections["arrival_delay_min_corrected"].min()),
                 float(corrections["arrival_delay_min_corrected"].max())]
                if len(corrections) else None
            ),
            "corrections_file": str(config.ROLLOVER_CORRECTIONS_FILE.name),
        },
        "delays_after_cleaning": {
            "observed_arrival_delays": int(delay.notna().sum()),
            "min": delay.min(),
            "max": delay.max(),
            "below_minus_60_min_kept_as_genuine_early_arrivals": int((delay < -60).sum()),
            f"above_{config.HIGH_DELAY_REVIEW_MIN}_min_unchanged_for_review": int(
                (delay > config.HIGH_DELAY_REVIEW_MIN).sum()
            ),
        },
        "structure": {
            "origin_rows": int(clean["is_origin"].sum()),
            "destination_rows": int(clean["is_destination"].sum()),
            "unscheduled_stop_rows": int(clean["is_unscheduled_stop"].sum()),
            "non_origin_rows_without_scheduled_arrival": int(
                (clean["scheduled_arrival"].isna() & ~clean["is_origin"]).sum()
            ),
            "journeys_with_repeated_station": int(repeated.groupby(key).ngroups),
            "rows_in_repeated_station_journeys": int(len(repeated)),
            "journeys_with_non_contiguous_seq": non_contiguous,
        },
        "completeness": {
            "note": (
                "status describes the journey, not every station. Missing actual times "
                "are NOT filled in; they stay missing."
            ),
            "by_status": by_status,
            "completed_journeys": int(len(completed)),
            "completed_journeys_with_every_scheduled_arrival_observed": int(completed["all"].sum()),
            "completed_stops_without_actual_arrival": int(
                (~scheduled_stops[scheduled_stops["status"] == "completed"]["arrival_observed"]).sum()
            ),
        },
        "coverage": {
            "rows_by_start_date": {str(k): int(v) for k, v in rows_by_date.items()},
            "last_start_date_is_partial": partial_last_day,
            "note": (
                "The source file ends at the snapshot date, so journeys starting on the "
                "last date are truncated. Do not treat the last date as a full day."
            ),
        },
        "validation_checks": checks,
    }
    return to_jsonable(report)


def run(
    raw_path=None,
    out_path=None,
    report_path=None,
    corrections_path=None,
    reader=None,
    writer=None,
):
    """Read raw data, clean it, validate it, and write outputs (backing up old ones)."""
    raw_path = Path(raw_path or config.RAW_RAILWAY_FILE)
    out_path = Path(out_path or config.CLEAN_RAILWAY_FILE)
    report_path = Path(report_path or config.QUALITY_REPORT_FILE)
    corrections_path = Path(corrections_path or config.ROLLOVER_CORRECTIONS_FILE)
    reader = reader or pd.read_parquet
    writer = writer or (lambda frame, path: frame.to_parquet(path, index=False))

    raw = reader(raw_path)
    clean, corrections, unresolved = clean_railway_data(raw)
    checks = validate_clean(clean, raw_rows=len(raw))
    report = build_quality_report(raw, clean, corrections, unresolved, checks)

    for path in (out_path, report_path, corrections_path):
        ensure_dir(path.parent)
        backup_file(path)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    corrections.to_csv(corrections_path, index=False)

    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise DataValidationError(
            "Cleaned data failed validation (report written for diagnosis): "
            + "; ".join(f"{c['name']} ({c['detail']})" for c in failed)
        )

    writer(clean, out_path)
    return clean, report


def main():
    parser = argparse.ArgumentParser(description="Clean the raw RailPulse railway data.")
    parser.add_argument("--raw", default=str(config.RAW_RAILWAY_FILE))
    parser.add_argument("--out", default=str(config.CLEAN_RAILWAY_FILE))
    args = parser.parse_args()

    clean, report = run(raw_path=args.raw, out_path=args.out)
    roll = report["midnight_rollover"]
    print("Cleaning completed.")
    print(f"  Rows: {len(clean):,} (none dropped)")
    print(f"  Midnight rollover rows corrected: {roll['corrected_rows']} (unresolved: {roll['unresolved_rows']})")
    print(f"  Saved: {args.out}")
    print(f"  Report: {config.QUALITY_REPORT_FILE}")
    print("  All validation checks passed.")


if __name__ == "__main__":
    main()
