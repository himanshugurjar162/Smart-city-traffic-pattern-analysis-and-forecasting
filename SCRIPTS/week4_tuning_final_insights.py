"""
Week 4 - Model Tuning, Final Comparison & Infrastructure Insights
--------------------------------------------------------------------
Run this after Week 3's script (needs data/traffic_data_cleaned.csv).

What this script does (matches Week 4 report):
  1. Tunes the Random Forest model per junction using a small grid search
     over n_estimators/max_depth, validated on a held-out slice of the
     training period (never touching the final test set until the end).
  2. Compares the tuned Random Forest against the Week 3 baseline models
     (Linear Regression, untuned Random Forest) on the same test set.
  3. Extracts feature importances from the tuned model to understand
     which factors (hour, weekend, holiday, etc.) matter most for each
     junction - this directly informs infrastructure planning.
  4. Builds a peak-hour summary per junction to support concrete
     recommendations (e.g., where/when extra traffic management is
     needed).
  5. Saves all results/plots as screenshots for the final report.
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

PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 6},
    {"n_estimators": 150, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 14},
    {"n_estimators": 200, "max_depth": None},
]


def save_dataframe_as_image(df, title, filename, figsize=None):
    if figsize is None:
        figsize = (9.5, 0.5 * (len(df) + 2))
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


def chronological_split(df, frac):
    df = df.sort_values("DateTime")
    split_idx = int(len(df) * (1 - frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def tune_random_forest(train_full):
    """Splits train_full into train/validation (chronologically) and picks
    the best parameter combination by validation MAE."""
    train, val = chronological_split(train_full, frac=0.15)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    best_params, best_mae = None, np.inf
    for params in PARAM_GRID:
        model = RandomForestRegressor(random_state=42, **params)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        if mae < best_mae:
            best_mae, best_params = mae, params

    return best_params


def run_final_comparison(df):
    results = []
    importances_by_junction = {}

    for junction in sorted(df["Junction"].unique()):
        jdf = df[df["Junction"] == junction].copy()
        train_full, test = chronological_split(jdf, frac=0.15)
        X_train, y_train = train_full[FEATURES], train_full[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]

        # --- Baseline: Linear Regression ---
        lr = LinearRegression().fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        results.append({
            "Junction": junction, "Model": "Linear Regression",
            "MAE": round(mean_absolute_error(y_test, lr_preds), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, lr_preds)), 2),
        })

        # --- Baseline: Untuned Random Forest (Week 3 settings) ---
        rf_base = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        rf_base.fit(X_train, y_train)
        rf_base_preds = rf_base.predict(X_test)
        results.append({
            "Junction": junction, "Model": "Random Forest (Week 3)",
            "MAE": round(mean_absolute_error(y_test, rf_base_preds), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, rf_base_preds)), 2),
        })

        # --- Tuned Random Forest ---
        best_params = tune_random_forest(train_full)
        rf_tuned = RandomForestRegressor(random_state=42, **best_params)
        rf_tuned.fit(X_train, y_train)
        rf_tuned_preds = rf_tuned.predict(X_test)
        results.append({
            "Junction": junction, "Model": f"Random Forest (Tuned: {best_params})",
            "MAE": round(mean_absolute_error(y_test, rf_tuned_preds), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, rf_tuned_preds)), 2),
        })

        importances_by_junction[junction] = pd.Series(rf_tuned.feature_importances_, index=FEATURES)

    return pd.DataFrame(results), importances_by_junction


def plot_feature_importance(importances_by_junction):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, (junction, importances) in zip(axes, sorted(importances_by_junction.items())):
        importances.sort_values().plot(kind="barh", ax=ax, color="#16a085")
        ax.set_title(f"Junction {junction}")
        ax.set_xlabel("Importance")
    fig.suptitle("Feature Importance (Tuned Random Forest) - What Drives Traffic at Each Junction",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/week4_feature_importance.png", dpi=150)
    plt.close()
    print("Saved screenshot: week4_feature_importance.png")


def plot_final_mae_comparison(results_df):
    pivot = results_df.pivot_table(index="Junction", columns="Model", values="MAE",
                                    aggfunc="first")
    # Shorten tuned-model column name for a cleaner chart
    pivot.columns = ["Linear Regression" if "Linear" in c else
                     "Random Forest (Week 3)" if "Week 3" in c else
                     "Random Forest (Tuned)" for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, color=["#95a5a6", "#e67e22", "#27ae60"])
    ax.set_title("Final Model Comparison: MAE by Junction (Lower is Better)")
    ax.set_xlabel("Junction")
    ax.set_ylabel("MAE")
    ax.set_xticklabels(pivot.index, rotation=0)
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/week4_final_mae_comparison.png", dpi=150)
    plt.close()
    print("Saved screenshot: week4_final_mae_comparison.png")


def peak_hour_summary(df):
    summary = (df.groupby(["Junction", "Hour"])["Vehicles"].mean()
               .reset_index()
               .sort_values(["Junction", "Vehicles"], ascending=[True, False]))
    top2 = summary.groupby("Junction").head(2)
    top2 = top2.rename(columns={"Vehicles": "Avg Vehicles"})
    top2["Avg Vehicles"] = top2["Avg Vehicles"].round(1)
    return top2.reset_index(drop=True)


def main():
    print("=" * 60)
    print("WEEK 4: Model Tuning, Final Comparison & Infrastructure Insights")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])

    print("\n--- Step 1: Tuning Random Forest per Junction ---")
    print("Trying parameter combinations:", PARAM_GRID)
    results_df, importances_by_junction = run_final_comparison(df)

    print("\nFinal comparison (Linear Regression vs Week 3 RF vs Tuned RF):")
    print(results_df.to_string(index=False))
    save_dataframe_as_image(results_df, "Final Model Comparison - All Junctions",
                             "week4_final_results_table.png", figsize=(11, 3.2))

    print("\n--- Step 2: Visual Comparison & Feature Importance ---")
    plot_final_mae_comparison(results_df)
    plot_feature_importance(importances_by_junction)

    print("\n--- Step 3: Peak-Hour Summary for Infrastructure Planning ---")
    top2 = peak_hour_summary(df)
    print(top2.to_string(index=False))
    save_dataframe_as_image(top2, "Top 2 Peak Traffic Hours per Junction",
                             "week4_peak_hours_table.png", figsize=(6, 3.2))

    print("\nWeek 4 processing complete. Screenshots saved to outputs/screenshots/")


if __name__ == "__main__":
    main()
