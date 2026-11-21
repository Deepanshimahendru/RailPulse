"""Tests for the Phase 1 data-cleaning step (src/data_preprocessing.py)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src import data_preprocessing as dp
from src.utils import config
from src.utils.helpers import backup_file

IST = "Asia/Kolkata"


def ts(text):
    return pd.Timestamp(text, tz=IST) if text else pd.NaT


def make_frame(rows):
    """rows: (train, start_date, seq, station, sched_arr, sched_dep, act_arr, act_dep, status)."""
    records = []
    for train, date, seq, station, sa, sd, aa, ad, status in rows:
        records.append(
            {
                "train_number": train,
                "start_date": pd.Timestamp(date).date(),
                "seq": seq,
                "station_code": station,
                "scheduled_arrival": ts(sa),
                "scheduled_departure": ts(sd),
                "actual_arrival": ts(aa),
                "actual_departure": ts(ad),
                "status": status,
                # stored (possibly wrong) delays as in the raw source
                "arrival_delay_min": float("nan"),
                "departure_delay_min": float("nan"),
            }
        )
    frame = pd.DataFrame(records)
    for column in dp.TIME_COLUMNS:
        frame[column] = pd.to_datetime(frame[column])
    return frame


def rollover_journey():
    """Journey where stop 2 has the arrival stamped with the next day's date."""
    rows = [
        ("04090", "2026-10-07", 1, "NDLS", None, "2026-10-07 22:00", None, "2026-10-07 22:05", "completed"),
        # arrival 23:55 wrongly dated 10-08; departure 00:05 on 10-08; actual arrival 00:29 on 10-08
        ("04090", "2026-10-07", 2, "DLI", "2026-10-08 23:55", "2026-10-08 00:05",
         "2026-10-08 00:29", "2026-10-08 00:55", "completed"),
        ("04090", "2026-10-07", 3, "MTC", "2026-10-08 02:00", "2026-10-08 02:05",
         "2026-10-08 02:30", "2026-10-08 02:35", "completed"),
    ]
    frame = make_frame(rows)
    frame.loc[1, "arrival_delay_min"] = -1406.0  # the bad value stored by the source
    return frame


# ---------------------------------------------------------------- rollover
def test_rollover_arrival_is_moved_back_one_day_and_delay_corrected():
    clean, corrections, unresolved = dp.clean_railway_data(rollover_journey())
    row = clean[clean["station_code"] == "DLI"].iloc[0]
    assert row["scheduled_arrival"] == ts("2026-10-07 23:55")
    assert row["arrival_delay_min"] == 34.0
    assert bool(row["rollover_corrected"]) is True
    assert len(corrections) == 1 and unresolved.empty
    assert corrections.iloc[0]["arrival_delay_min_original"] == -1406.0
    assert corrections.iloc[0]["arrival_delay_min_corrected"] == 34.0


def test_rollover_not_applied_when_correction_breaks_journey_order():
    frame = rollover_journey()
    # previous stop now departs AFTER the corrected arrival, so the shift is not justified
    frame.loc[0, "scheduled_departure"] = ts("2026-10-08 00:00")
    clean, corrections, unresolved = dp.clean_railway_data(frame)
    assert corrections.empty
    assert len(unresolved) == 1
    row = clean[clean["station_code"] == "DLI"].iloc[0]
    assert row["scheduled_arrival"] == ts("2026-10-08 23:55")  # left unchanged


def test_normal_midnight_crossing_is_not_touched():
    rows = [
        ("11111", "2026-10-07", 1, "AAA", None, "2026-10-07 23:30", None, "2026-10-07 23:35", "completed"),
        ("11111", "2026-10-07", 2, "BBB", "2026-10-08 00:20", "2026-10-08 00:25",
         "2026-10-08 00:40", "2026-10-08 00:45", "completed"),
    ]
    clean, corrections, unresolved = dp.clean_railway_data(make_frame(rows))
    assert corrections.empty and unresolved.empty
    assert clean.iloc[1]["arrival_delay_min"] == 20.0


# ---------------------------------------------------------------- delays
def test_missing_actual_times_stay_missing_not_imputed():
    rows = [
        ("22222", "2026-10-07", 1, "AAA", None, "2026-10-07 10:00", None, "2026-10-07 10:00", "completed"),
        ("22222", "2026-10-07", 2, "BBB", "2026-10-07 11:00", "2026-10-07 11:02", None, None, "completed"),
        ("22222", "2026-10-07", 3, "CCC", "2026-10-07 12:00", None, "2026-10-07 12:10", None, "completed"),
    ]
    clean, _, _ = dp.clean_railway_data(make_frame(rows))
    assert pd.isna(clean.loc[1, "arrival_delay_min"])
    assert bool(clean.loc[1, "arrival_observed"]) is False
    assert clean.loc[2, "arrival_delay_min"] == 10.0
    assert pd.isna(clean.loc[0, "arrival_delay_min"])  # origin has no arrival


# ---------------------------------------------------------------- structure
def test_structure_flags_and_no_rows_dropped():
    rows = [
        ("33333", "2026-10-07", 1, "AAA", None, "2026-10-07 08:00", None, "2026-10-07 08:00", "diverted"),
        ("33333", "2026-10-07", 2, "BBB", "2026-10-07 09:00", "2026-10-07 09:02", "2026-10-07 09:05", None, "diverted"),
        ("33333", "2026-10-07", 3, "AAA", "2026-10-07 10:00", None, "2026-10-07 10:03", None, "diverted"),
        ("33333", "2026-10-07", 1013, "ZZZ", None, None, "2026-10-07 11:00", None, "diverted"),
    ]
    frame = make_frame(rows)
    clean, _, _ = dp.clean_railway_data(frame)
    assert len(clean) == len(frame)
    assert clean["is_origin"].tolist() == [True, False, False, False]
    assert clean["is_destination"].tolist() == [False, False, True, False]
    assert clean["is_unscheduled_stop"].tolist() == [False, False, False, True]
    assert clean["route_pos"].tolist()[:3] == [1, 2, 3] and pd.isna(clean["route_pos"].iloc[3])
    assert clean["station_repeated_in_journey"].tolist() == [True, False, True, False]
    assert clean["station_visit_no"].tolist() == [1, 1, 2, 1]


def test_train_number_keeps_leading_zeros():
    clean, _, _ = dp.clean_railway_data(rollover_journey())
    assert set(clean["train_number"]) == {"04090"}


# ---------------------------------------------------------------- validation
def test_validation_passes_after_cleaning_and_fails_on_injected_error():
    frame = rollover_journey()
    clean, _, _ = dp.clean_railway_data(frame)
    assert all(c["passed"] for c in dp.validate_clean(clean, raw_rows=len(frame)))

    broken = clean.copy()
    broken.loc[1, "scheduled_arrival"] = ts("2026-10-08 23:55")
    failed = {c["name"] for c in dp.validate_clean(broken) if not c["passed"]}
    assert "scheduled_arrival_not_after_departure" in failed


def test_cleaning_is_idempotent():
    once, _, _ = dp.clean_railway_data(rollover_journey())
    twice, corrections, _ = dp.clean_railway_data(once)
    assert corrections.empty
    pd.testing.assert_series_equal(once["arrival_delay_min"], twice["arrival_delay_min"])


def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        dp.standardise(rollover_journey().drop(columns=["status"]))


# ---------------------------------------------------------------- backups / run
def test_backup_file_copies_and_never_overwrites(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("a,b\n1,2\n")
    first = backup_file(source, tmp_path / "bk")
    second = backup_file(source, tmp_path / "bk")
    assert first.read_text() == "a,b\n1,2\n"
    assert first != second
    assert backup_file(tmp_path / "missing.csv", tmp_path / "bk") is None


def test_run_writes_report_and_backs_up_existing_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BACKUP_DIR", tmp_path / "bk")
    out = tmp_path / "clean.pkl"
    out.write_text("old output")
    report = tmp_path / "report.json"
    corrections = tmp_path / "corrections.csv"
    clean, rep = dp.run(
        raw_path=tmp_path / "ignored",
        out_path=out,
        report_path=report,
        corrections_path=corrections,
        reader=lambda _p: rollover_journey(),
        writer=lambda frame, path: frame.to_pickle(path),
    )
    assert rep["midnight_rollover"]["corrected_rows"] == 1
    assert json.loads(report.read_text())["midnight_rollover"]["unresolved_rows"] == 0
    assert list((tmp_path / "bk").glob("clean_*.pkl"))  # old output was backed up
    assert pd.read_csv(corrections).shape[0] == 1


# ---------------------------------------------------------------- real data
RAW = Path(config.RAW_RAILWAY_FILE)


@pytest.mark.skipif(not RAW.exists(), reason="raw parquet not present")
def test_real_raw_file_cleans_and_validates():
    try:
        raw = pd.read_parquet(RAW)
    except ImportError:
        pytest.skip("no parquet engine installed")
    clean, corrections, unresolved = dp.clean_railway_data(raw)
    assert len(clean) == len(raw)
    assert unresolved.empty
    assert all(c["passed"] for c in dp.validate_clean(clean, raw_rows=len(raw)))
