"""
Week 3 - Model Building & Evaluation
--------------------------------------
Run this after Week 2's script (needs data/traffic_data_cleaned.csv).

What this script does (matches Week 3 report):
  1. Builds simple features (hour, day of week, month, weekend flag,
     holiday flag) for each junction.
  2. Splits each junction's data chronologically into train/test sets
     (last 15% of the timeline held out as test - a realistic approach
     for time-series data, since random splitting would leak the future
     into training).
  3. Trains two basic models per junction: Linear Regression and
     Random Forest Regressor.
  4. Evaluates both models using MAE and RMSE.
  5. Saves a results table and comparison/actual-vs-predicted plots as
     screenshots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/traffic_data_cleaned.csv"
SCREENSHOT_DIR = "outputs/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

FEATURES = ["Hour", "DayOfWeek", "Month", "IsWeekend", "IsHoliday"]
TARGET = "Vehicles"


def save_dataframe_as_image(df, title, filename, figsize=None):
    if figsize is None:
        figsize = (9, 0.5 * (len(df) + 2))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
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


def chronological_split(df, test_frac=0.15):
    df = df.sort_values("DateTime")
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_and_evaluate(df):
    results = []
    predictions_for_plot = {}

    for junction in sorted(df["Junction"].unique()):
        jdf = df[df["Junction"] == junction].copy()
        train, test = chronological_split(jdf)

        X_train, y_train = train[FEATURES], train[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            results.append({
                "Junction": junction,
                "Model": name,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
            })
            if name == "Random Forest":
                predictions_for_plot[junction] = (test["DateTime"].values, y_test.values, preds)

    return pd.DataFrame(results), predictions_for_plot


def plot_model_comparison(results_df):
    pivot_mae = results_df.pivot(index="Junction", columns="Model", values="MAE")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_mae.plot(kind="bar", ax=ax, color=["#3498db", "#e67e22"])
    ax.set_title("Model Comparison: Mean Absolute Error (MAE) by Junction")
    ax.set_xlabel("Junction")
    ax.set_ylabel("MAE (lower is better)")
    ax.set_xticklabels(pivot_mae.index, rotation=0)
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/week3_model_comparison_mae.png", dpi=150)
    plt.close()
    print("Saved screenshot: week3_model_comparison_mae.png")


def plot_actual_vs_predicted(predictions_for_plot, junction=1, n_points=150):
    dates, actual, predicted = predictions_for_plot[junction]
    dates, actual, predicted = dates[:n_points], actual[:n_points], predicted[:n_points]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, actual, label="Actual", color="#2c3e50", linewidth=1.5)
    ax.plot(dates, predicted, label="Predicted (Random Forest)", color="#e74c3c",
            linewidth=1.5, linestyle="--")
    ax.set_title(f"Actual vs Predicted Traffic - Junction {junction} (first {n_points} test hours)")
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Vehicle Count")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/week3_actual_vs_predicted_junction{junction}.png", dpi=150)
    plt.close()
    print(f"Saved screenshot: week3_actual_vs_predicted_junction{junction}.png")


def main():
    print("=" * 60)
    print("WEEK 3: Model Building & Evaluation")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])

    print("\n--- Step 1: Chronological Train/Test Split & Model Training ---")
    print("Training Linear Regression and Random Forest for each junction...")
    results_df, predictions_for_plot = train_and_evaluate(df)

    print("\nModel evaluation results (MAE & RMSE per junction):")
    print(results_df.to_string(index=False))

    save_dataframe_as_image(results_df, "Model Evaluation Results (MAE & RMSE per Junction)",
                             "week3_results_table.png")

    print("\n--- Step 2: Visual Comparison ---")
    plot_model_comparison(results_df)
    plot_actual_vs_predicted(predictions_for_plot, junction=1)

    best_model_overall = results_df.groupby("Model")["MAE"].mean().idxmin()
    print(f"\nOverall, the model with the lowest average MAE across junctions: {best_model_overall}")

    print("\nWeek 3 model building complete. Screenshots saved to outputs/screenshots/")


if __name__ == "__main__":
    main()
