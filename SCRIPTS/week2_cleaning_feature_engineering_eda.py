"""
Week 2 - Data Cleaning, Feature Engineering & EDA
---------------------------------------------------
Run this after Week 1's script. It produces a cleaned CSV
(data/traffic_data_cleaned.csv) and a set of EDA screenshots in
outputs/screenshots/ that map directly to the Week 2 report sections.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/traffic_data.csv"
CLEANED_PATH = "data/traffic_data_cleaned.csv"
SCREENSHOT_DIR = "outputs/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

sns.set_style("whitegrid")

# Same illustrative holiday list used to generate the data (in your own
# project, replace this with the real holiday calendar for your city/year).
HOLIDAYS = pd.to_datetime([
    "2015-11-11", "2015-11-25", "2015-12-25", "2016-01-01", "2016-01-26",
    "2016-03-24", "2016-08-15", "2016-08-25", "2016-10-02", "2016-10-11",
    "2016-10-30", "2016-12-25", "2017-01-01", "2017-01-26", "2017-03-13",
    "2017-08-15", "2017-10-02",
])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)

    # Fill missing Vehicles counts using the median for that junction & hour
    # (a simple, explainable approach appropriate for a student project).
    df["Hour"] = df["DateTime"].dt.hour
    df["Vehicles"] = df.groupby(["Junction", "Hour"])["Vehicles"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Vehicles"] = df["Vehicles"].round().astype(int)

    print(f"Removed {before - after_dedup} duplicate rows.")
    print(f"Filled remaining missing values using junction+hour median.")
    print(f"Remaining missing values: {df['Vehicles'].isna().sum()}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Day"] = df["DateTime"].dt.day
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["DayName"] = df["DateTime"].dt.day_name()
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsHoliday"] = df["DateTime"].dt.normalize().isin(HOLIDAYS).astype(int)

    def time_of_day(h):
        if 5 <= h < 12:
            return "Morning"
        elif 12 <= h < 17:
            return "Afternoon"
        elif 17 <= h < 21:
            return "Evening"
        return "Night"

    df["TimeOfDay"] = df["Hour"].apply(time_of_day)
    return df


def plot_hourly_pattern(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    for j in sorted(df["Junction"].unique()):
        subset = df[df["Junction"] == j].groupby("Hour")["Vehicles"].mean()
        ax.plot(subset.index, subset.values, marker="o", label=f"Junction {j}")
    ax.set_title("Average Hourly Traffic Pattern by Junction")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Vehicle Count")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/week2_hourly_pattern.png", dpi=150)
    plt.close()
    print("Saved screenshot: week2_hourly_pattern.png")


def plot_weekday_vs_weekend(df):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=False)
    for ax, j in zip(axes, sorted(df["Junction"].unique())):
        subset = df[df["Junction"] == j]
        sns.boxplot(data=subset, x="IsWeekend", y="Vehicles", ax=ax,
                    hue="IsWeekend", palette=["#3498db", "#e67e22"], legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Weekday", "Weekend"])
        ax.set_title(f"Junction {j}")
        ax.set_xlabel("")
    fig.suptitle("Weekday vs Weekend Traffic Distribution by Junction", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/week2_weekday_vs_weekend.png", dpi=150)
    plt.close()
    print("Saved screenshot: week2_weekday_vs_weekend.png")


def plot_holiday_vs_normal(df):
    summary = df.groupby(["Junction", "IsHoliday"])["Vehicles"].mean().unstack()
    summary.columns = ["Normal Day", "Holiday/Special Occasion"]
    fig, ax = plt.subplots(figsize=(8, 5))
    summary.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Average Traffic: Normal Days vs Holidays/Special Occasions")
    ax.set_xlabel("Junction")
    ax.set_ylabel("Average Vehicle Count")
    ax.set_xticklabels(summary.index, rotation=0)
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/week2_holiday_vs_normal.png", dpi=150)
    plt.close()
    print("Saved screenshot: week2_holiday_vs_normal.png")


def plot_overall_trend(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    daily = df.groupby([df["DateTime"].dt.date, "Junction"])["Vehicles"].sum().reset_index()
    for j in sorted(df["Junction"].unique()):
        subset = daily[daily["Junction"] == j]
        ax.plot(subset["DateTime"], subset["Vehicles"], label=f"Junction {j}", linewidth=0.8)
    ax.set_title("Daily Total Traffic Trend by Junction (Nov 2015 - Jun 2017)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Daily Vehicle Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/week2_overall_trend.png", dpi=150)
    plt.close()
    print("Saved screenshot: week2_overall_trend.png")


def main():
    print("=" * 60)
    print("WEEK 2: Cleaning, Feature Engineering & EDA")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])

    print("\n--- Step 1: Data Cleaning ---")
    df = clean_data(df)

    print("\n--- Step 2: Feature Engineering ---")
    df = add_features(df)
    print("New columns added:", [c for c in df.columns if c not in
          ["DateTime", "Junction", "Vehicles", "ID"]])

    df.to_csv(CLEANED_PATH, index=False)
    print(f"\nCleaned & feature-engineered data saved to {CLEANED_PATH}")

    print("\n--- Step 3: Exploratory Data Analysis ---")
    plot_hourly_pattern(df)
    plot_weekday_vs_weekend(df)
    plot_holiday_vs_normal(df)
    plot_overall_trend(df)

    print("\nWeek 2 processing complete. Screenshots saved to outputs/screenshots/")


if __name__ == "__main__":
    main()
