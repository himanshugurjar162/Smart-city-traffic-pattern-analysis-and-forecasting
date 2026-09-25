"""
generate_traffic_data.py
-------------------------
Generates a realistic hourly traffic-count dataset for 4 city junctions,
in the same format as the popular Kaggle "Traffic Prediction Dataset"
(columns: DateTime, Junction, Vehicles, ID).

Why this exists:
Internet access isn't guaranteed on every machine/grader, so this script
lets the whole project run end-to-end offline with realistic synthetic
data. If you have the original Kaggle dataset (search "Traffic Prediction
Dataset" by fedesoriano on kaggle.com), just drop your traffic.csv into
the data/ folder and skip this script — the rest of the pipeline reads
whichever CSV is named traffic_data.csv.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

START_DATE = "2015-11-01"
END_DATE = "2017-06-30 23:00:00"

# Base traffic level and weekly/holiday sensitivity differ per junction,
# just like real junctions differ by road size / location.
JUNCTION_PROFILES = {
    1: {"base": 60, "peak_boost": 45, "weekend_drop": 0.25, "holiday_drop": 0.35},
    2: {"base": 25, "peak_boost": 15, "weekend_drop": 0.15, "holiday_drop": 0.20},
    3: {"base": 15, "peak_boost": 8,  "weekend_drop": 0.10, "holiday_drop": 0.15},
    4: {"base": 8,  "peak_boost": 5,  "weekend_drop": 0.05, "holiday_drop": 0.10},
}

# A representative set of Indian public holidays / festive occasions
# across the date range (kept short and illustrative for a student project).
HOLIDAYS = pd.to_datetime([
    "2015-11-11", "2015-11-25", "2015-12-25", "2016-01-01", "2016-01-26",
    "2016-03-24", "2016-08-15", "2016-08-25", "2016-10-02", "2016-10-11",
    "2016-10-30", "2016-12-25", "2017-01-01", "2017-01-26", "2017-03-13",
    "2017-08-15", "2017-10-02",
])


def hour_factor(hour: int) -> float:
    """Typical daily traffic curve: low at night, peaks at ~9am and ~6pm."""
    morning_peak = np.exp(-((hour - 9) ** 2) / (2 * 2.2 ** 2))
    evening_peak = np.exp(-((hour - 18) ** 2) / (2 * 2.5 ** 2))
    base_curve = 0.15 + 0.55 * (morning_peak + evening_peak)
    return base_curve


def generate_junction_series(junction_id: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    profile = JUNCTION_PROFILES[junction_id]
    hours = dates.hour
    is_weekend = dates.dayofweek >= 5
    is_holiday = dates.normalize().isin(HOLIDAYS)

    # Slow year-over-year growth trend (city traffic increasing over time)
    days_elapsed = (dates - dates[0]).days
    growth = 1 + (days_elapsed / days_elapsed.max()) * 0.35

    daily_shape = np.array([hour_factor(h) for h in hours])
    vehicles = profile["base"] * growth * (0.5 + daily_shape * profile["peak_boost"] / profile["base"])

    vehicles = np.where(is_weekend, vehicles * (1 - profile["weekend_drop"]), vehicles)
    vehicles = np.where(is_holiday, vehicles * (1 - profile["holiday_drop"]), vehicles)

    noise = np.random.normal(0, profile["base"] * 0.08, size=len(dates))
    vehicles = np.clip(vehicles + noise, 1, None).round().astype(int)

    return pd.DataFrame({"DateTime": dates, "Junction": junction_id, "Vehicles": vehicles})


def main():
    dates = pd.date_range(START_DATE, END_DATE, freq="h")
    frames = [generate_junction_series(j, dates) for j in JUNCTION_PROFILES]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["DateTime", "Junction"]).reset_index(drop=True)
    df["ID"] = (df["DateTime"].dt.strftime("%Y%m%d%H").astype(str) + df["Junction"].astype(str)).astype(np.int64)
    df = df[["DateTime", "Junction", "Vehicles", "ID"]]

    # Introduce a small number of missing values and duplicates on purpose,
    # so the Week 1/2 cleaning steps have something real to do.
    missing_idx = np.random.choice(df.index, size=25, replace=False)
    df.loc[missing_idx, "Vehicles"] = np.nan
    dup_rows = df.sample(10, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    out_path = "data/traffic_data.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
