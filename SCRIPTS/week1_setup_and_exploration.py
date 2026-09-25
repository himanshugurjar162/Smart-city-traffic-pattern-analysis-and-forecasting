"""
Week 1 - Setup and Initial Data Exploration
--------------------------------------------
Run this after data/generate_traffic_data.py (or after placing your own
traffic_data.csv in the data/ folder).

What this script does (matches Week 1 report):
  1. Loads the traffic dataset for the 4 junctions.
  2. Inspects its structure (shape, dtypes, sample rows).
  3. Checks for missing values and duplicate rows.
  4. Saves a few screenshots (as PNG images) into outputs/screenshots/
     so they can be dropped straight into the weekly report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_PATH = "data/traffic_data.csv"
SCREENSHOT_DIR = "outputs/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def save_dataframe_as_image(df, title, filename, col_widths=None):
    """Renders a DataFrame as a clean table image (a 'screenshot' of the data)."""
    fig, ax = plt.subplots(figsize=(9, 0.45 * (len(df) + 2)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f5f6fa" if row % 2 == 0 else "white")
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved screenshot: {SCREENSHOT_DIR}/{filename}")


def save_text_as_image(text, title, filename, figsize=(8, 3.5)):
    """Renders console-style text (like df.info()) as an image."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, loc="left")
    ax.text(0.01, 0.98, text, family="monospace", fontsize=9,
            va="top", ha="left", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved screenshot: {SCREENSHOT_DIR}/{filename}")


def main():
    print("=" * 60)
    print("WEEK 1: Loading and Exploring the Traffic Dataset")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])

    print(f"\nDataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Junctions present: {sorted(df['Junction'].unique())}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")

    print("\nFirst 5 rows:")
    print(df.head())
    save_dataframe_as_image(df.head(8), "Traffic Dataset - Sample Rows (df.head())",
                             "week1_dataframe_head.png")

    missing = df.isna().sum()
    duplicates = df.duplicated().sum()
    print("\nMissing values per column:")
    print(missing)
    print(f"\nDuplicate rows: {duplicates}")

    info_text = (
        f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        f"Column dtypes:\n"
        f"{df.dtypes.to_string()}\n\n"
        f"Missing values:\n"
        f"{missing.to_string()}\n\n"
        f"Duplicate rows: {duplicates}"
    )
    save_text_as_image(info_text, "Dataset Structure & Data Quality Check",
                        "week1_data_info.png", figsize=(7, 5))

    counts_per_junction = df.groupby("Junction")["Vehicles"].count()
    print("\nRecord count per junction:")
    print(counts_per_junction)

    fig, ax = plt.subplots(figsize=(7, 4))
    counts_per_junction.plot(kind="bar", ax=ax, color=["#3498db", "#e67e22", "#2ecc71", "#e74c3c"])
    ax.set_title("Number of Records per Junction")
    ax.set_xlabel("Junction")
    ax.set_ylabel("Record Count")
    ax.set_xticklabels(counts_per_junction.index, rotation=0)
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/week1_records_per_junction.png", dpi=150)
    plt.close()
    print(f"Saved screenshot: {SCREENSHOT_DIR}/week1_records_per_junction.png")

    print("\nWeek 1 exploration complete. Screenshots saved to outputs/screenshots/")


if __name__ == "__main__":
    main()
