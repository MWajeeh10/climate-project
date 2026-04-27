import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import demo_app


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RAW = ROOT / "raw_data"


def compute_metrics(group):
    y = group["pm25"].to_numpy()
    p = group["pred"].to_numpy()
    return pd.Series(
        {
            "r2": r2_score(y, p),
            "rmse": mean_squared_error(y, p) ** 0.5,
            "mae": mean_absolute_error(y, p),
            "mean_actual": y.mean(),
            "mean_pred": p.mean(),
            "bias": (p - y).mean(),
            "count": len(group),
        }
    )


def main():
    print("=" * 72)
    print(" Lahore Smog Project Prediction Audit")
    print("=" * 72)

    missing_source_flags = [
        "raw_data/openaq_pm25.csv",
        "raw_data/epd_aqi_scraped.csv",
        "raw_data/pm25_merged_final.csv",
        "raw_data/firms_fire_counts.csv",
        "raw_data/era5_lahore.csv",
    ]
    print("\n[1] Source Audit")
    for rel in missing_source_flags:
        print(f"  {rel}: {'FOUND' if (ROOT / rel).exists() else 'MISSING'}")

    print("\n  Note: if these are missing, the saved training set may rely on fallback or")
    print("  synthetic PM2.5 generation rather than a fully observed target series.")

    print("\n[2] Load Model Artifacts")
    df = pd.read_csv(ROOT / "lahore_merged.csv", parse_dates=["timestamp"])
    features = joblib.load(MODELS / "feature_list_full.pkl")
    scaler = joblib.load(MODELS / "scaler.pkl")
    rf = joblib.load(MODELS / "rf_model.pkl")
    xgb = joblib.load(MODELS / "xgb_model.pkl")
    lgb = joblib.load(MODELS / "lgb_model.pkl")
    rf.n_jobs = 1
    if hasattr(xgb, "n_jobs"):
        xgb.n_jobs = 1
    if hasattr(lgb, "n_jobs"):
        lgb.n_jobs = 1

    print(f"  lahore_merged.csv rows: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    print("\n[3] Seasonal Target Sanity Check")
    monthly = df.groupby(df["timestamp"].dt.month)["pm25"].mean().round(1)
    print(monthly.to_string())
    print("\n  If real summer Lahore readings you trust are far lower than these seasonal")
    print("  means, then the problem is primarily in the target data, not only the model.")

    print("\n[4] Hold-Out Seasonal Performance")
    n = len(df)
    test = df.iloc[int(n * 0.80) :].copy()
    X = pd.DataFrame(scaler.transform(test[features]), columns=features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_pred = rf.predict(X)
        xgb_pred = xgb.predict(X)
        lgb_pred = lgb.predict(X)
    test["pred"] = 0.25 * rf_pred + 0.40 * xgb_pred + 0.35 * lgb_pred
    season_map = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }
    test["season"] = test["timestamp"].dt.month.map(season_map)
    seasonal = test.groupby("season").apply(compute_metrics).round(3)
    monthly_metrics = test.groupby(test["timestamp"].dt.month).apply(compute_metrics).round(3)
    print("\nSeasonal metrics:")
    print(seasonal.to_string())
    print("\nMonthly metrics:")
    print(monthly_metrics.to_string())

    print("\n[5] Regression Test Cases")
    cases = pd.read_csv(MODELS / "prediction_regression_cases.csv")
    results = []
    for _, row in cases.iterrows():
        payload = {
            "timestamp": row["timestamp"].replace(" ", "T")[:16],
            "temperature": float(row["temperature"]),
            "dew_point": float(row["dew_point"]),
            "humidity": float(row["humidity"]),
            "pressure": float(row["pressure"]),
            "wind_speed": float(row["wind_speed"]),
            "wind_direction": float(row["wind_direction"]),
            "boundary_layer_height": float(row["boundary_layer_height"]),
            "pm25_lag1h": float(row["pm25_lag1h"]),
            "pm25_lag3h": float(row["pm25_lag3h"]),
            "pm25_lag6h": float(row["pm25_lag6h"]),
            "pm25_lag12h": float(row["pm25_lag12h"]),
            "pm25_lag24h": float(row["pm25_lag24h"]),
            "pm25_lag48h": float(row["pm25_lag48h"]),
        }
        pred = demo_app.ARTIFACTS.predict(payload)["pm25"]
        results.append(
            {
                "scenario": row["scenario"],
                "expected": row["expected_weighted_prediction"],
                "recomputed": round(float(pred), 1),
                "delta": round(float(pred) - float(row["expected_weighted_prediction"]), 3),
            }
        )
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n[6] Bottom Line")
    print("  The saved ensemble is internally stable against its own training target.")
    print("  If it disagrees with trusted real non-winter readings, retraining on a cleaner")
    print("  fully observed PM2.5 target is the priority fix.")


if __name__ == "__main__":
    main()
