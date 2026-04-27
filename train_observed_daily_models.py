"""
Train a cleaner daily PM2.5 model suite from lahore_air_quality_final_dataset.csv.

This script is intentionally separate from the legacy hourly pipeline because the
new dataset is daily, cleaner, and better suited for seasonal realism checks.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "lahore_air_quality_final_dataset.csv"
OUT_DIR = ROOT / "models_observed_daily"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["log_pm25_lag_1"] = np.log1p(df["pm25_lag_1"])
    df["log_pm25_lag_7"] = np.log1p(df["pm25_lag_7"])
    df["humidity_wind"] = df["RH2M"] / (df["WS2M"] + 0.5)
    df["temp_wind_ratio"] = df["T2M"] / (df["WS2M"] + 0.5)
    df["stagnation_proxy"] = np.clip(3.0 - df["WS2M"], 0, None) * (df["RH2M"] / 100)
    df["temp_sq"] = df["T2M"] ** 2
    df["lag_diff"] = df["pm25_lag_1"] - df["pm25_lag_7"]
    df["rolling_proxy"] = (df["pm25_lag_1"] + df["pm25_lag_7"]) / 2
    return df


FEATURES = [
    "T2M",
    "RH2M",
    "WS2M",
    "pm25_lag_1",
    "pm25_lag_7",
    "month",
    "smog_season",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "log_pm25_lag_1",
    "log_pm25_lag_7",
    "humidity_wind",
    "temp_wind_ratio",
    "stagnation_proxy",
    "temp_sq",
    "lag_diff",
    "rolling_proxy",
]


def metrics(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
    }


def main():
    print("=" * 72)
    print(" Lahore Observed Daily Model Training")
    print("=" * 72)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = engineer_features(df)
    print(f"Loaded {len(df):,} daily rows from {df['date'].min().date()} to {df['date'].max().date()}")

    train_end = int(len(df) * 0.80)
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]
    y_train = train_df["pm25"]
    y_test = test_df["pm25"]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    models = {
        "daily_random_forest": RandomForestRegressor(
            n_estimators=600,
            max_depth=18,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=SEED,
            n_jobs=1,
        ),
        "daily_ridge": Ridge(alpha=2.0),
        "daily_xgboost": xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        ),
        "daily_lightgbm": lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=5,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=1,
            verbosity=-1,
        ),
        "daily_extra_trees": ExtraTreesRegressor(
            n_estimators=700,
            max_depth=18,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=1,
        ),
    }

    rows = []
    preds = {}
    fitted_models = {}
    print("\nHold-out metrics:")
    for key, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xtr, y_train)
            pred = model.predict(Xte)
        preds[key] = pred
        fitted_models[key] = model
        row = {"model_key": key, **metrics(y_test, pred)}
        rows.append(row)
        print(
            f"  {key:20s} R2={row['r2']:.4f} "
            f"RMSE={row['rmse']:.2f} MAE={row['mae']:.2f}"
        )

    weighted_pred = (
        0.45 * preds["daily_random_forest"]
        + 0.20 * preds["daily_ridge"]
        + 0.15 * preds["daily_xgboost"]
        + 0.10 * preds["daily_lightgbm"]
        + 0.10 * preds["daily_extra_trees"]
    )
    weighted_row = {"model_key": "daily_weighted_ensemble", **metrics(y_test, weighted_pred)}
    rows.append(weighted_row)
    print(
        f"  {'daily_weighted_ensemble':20s} R2={weighted_row['r2']:.4f} "
        f"RMSE={weighted_row['rmse']:.2f} MAE={weighted_row['mae']:.2f}"
    )

    metrics_df = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(OUT_DIR / "daily_model_metrics.csv", index=False)

    # Refit on full data for deployment.
    X_full = df[FEATURES]
    y_full = df["pm25"]
    full_scaler = StandardScaler()
    Xfull_scaled = full_scaler.fit_transform(X_full)

    joblib.dump(full_scaler, OUT_DIR / "daily_scaler.pkl")
    joblib.dump(FEATURES, OUT_DIR / "daily_feature_list.pkl")
    joblib.dump({"train_rows": len(train_df), "test_rows": len(test_df)}, OUT_DIR / "daily_split_info.pkl")

    deployed = {}
    print("\nRefitting on full dataset for UI deployment:")
    for key, model in models.items():
        fresh = model.__class__(**model.get_params())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fresh.fit(Xfull_scaled, y_full)
        if hasattr(fresh, "n_jobs"):
            fresh.n_jobs = 1
        deployed[key] = fresh
        joblib.dump(fresh, OUT_DIR / f"{key}.pkl")
        print(f"  saved {key}.pkl")

    # Save a few comparison cases from the observed dataset.
    cases = df.iloc[[20, 95, 180, 320, 500, 620]].copy()
    case_rows = []
    X_cases = pd.DataFrame(full_scaler.transform(cases[FEATURES]), columns=FEATURES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_map = {
            "daily_random_forest": deployed["daily_random_forest"].predict(X_cases),
            "daily_ridge": deployed["daily_ridge"].predict(X_cases),
            "daily_xgboost": deployed["daily_xgboost"].predict(X_cases),
            "daily_lightgbm": deployed["daily_lightgbm"].predict(X_cases),
            "daily_extra_trees": deployed["daily_extra_trees"].predict(X_cases),
        }
    pred_map["daily_weighted_ensemble"] = (
        0.45 * pred_map["daily_random_forest"]
        + 0.20 * pred_map["daily_ridge"]
        + 0.15 * pred_map["daily_xgboost"]
        + 0.10 * pred_map["daily_lightgbm"]
        + 0.10 * pred_map["daily_extra_trees"]
    )

    for idx, (_, row) in enumerate(cases.iterrows(), start=1):
        case_rows.append(
            {
                "scenario": f"Observed Daily Case {idx}",
                "date": row["date"].strftime("%Y-%m-%d"),
                "actual_pm25": round(float(row["pm25"]), 1),
                "T2M": round(float(row["T2M"]), 2),
                "RH2M": round(float(row["RH2M"]), 2),
                "WS2M": round(float(row["WS2M"]), 2),
                "pm25_lag_1": round(float(row["pm25_lag_1"]), 2),
                "pm25_lag_7": round(float(row["pm25_lag_7"]), 2),
                "daily_random_forest": round(float(pred_map["daily_random_forest"][idx - 1]), 1),
                "daily_ridge": round(float(pred_map["daily_ridge"][idx - 1]), 1),
                "daily_xgboost": round(float(pred_map["daily_xgboost"][idx - 1]), 1),
                "daily_lightgbm": round(float(pred_map["daily_lightgbm"][idx - 1]), 1),
                "daily_extra_trees": round(float(pred_map["daily_extra_trees"][idx - 1]), 1),
                "daily_weighted_ensemble": round(float(pred_map["daily_weighted_ensemble"][idx - 1]), 1),
            }
        )
    pd.DataFrame(case_rows).to_csv(OUT_DIR / "daily_comparison_cases.csv", index=False)

    print("\nSaved:")
    print(f"  {OUT_DIR / 'daily_model_metrics.csv'}")
    print(f"  {OUT_DIR / 'daily_comparison_cases.csv'}")


if __name__ == "__main__":
    main()
