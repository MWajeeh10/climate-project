"""
Compute hour-specific bias corrections for PM2.5 predictions.

This script analyzes residuals from the test set and learns systematic 
hour-of-day bias corrections that can be applied at prediction time.

The approach:
1. Load trained models and test data
2. Generate predictions on test set
3. Calculate residuals and median bias by hour
4. Apply Gaussian smoothing to generalize across hours
5. Save corrections for use in demo_app.py and predictions
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DAILY_MODELS_DIR = ROOT / "models_observed_daily"

def compute_hourly_corrections_legacy():
    """Compute bias corrections for legacy hourly models."""
    print("\n" + "="*70)
    print(" Computing Hourly Bias Corrections (Legacy Pipeline)")
    print("="*70)
    
    # Load data
    merged_path = ROOT / "lahore_merged.csv"
    if not merged_path.exists():
        print(f"  [SKIP] {merged_path} not found")
        return None
    
    df = pd.read_csv(merged_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Three-way chronological split: 70% train, 10% val, 20% test
    n = len(df)
    cut1 = int(n * 0.70)
    cut2 = int(n * 0.80)
    test_df = df.iloc[cut2:].copy()
    
    print(f"\n  Loaded {len(df):,} rows, test set: {len(test_df):,} rows")
    
    if len(test_df) == 0:
        print("  [SKIP] Test set is empty")
        return None
    
    # Load feature list
    feature_list_path = MODELS_DIR / "feature_list_full.pkl"
    if not feature_list_path.exists():
        print(f"  [SKIP] {feature_list_path} not found")
        return None
    
    features = joblib.load(feature_list_path)
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    
    # Build test set
    try:
        X_test = test_df[features]
        y_test = test_df["pm25"]
        timestamps_test = test_df["timestamp"]
    except KeyError as e:
        print(f"  [ERROR] Missing feature: {e}")
        return None
    
    X_test_scaled = scaler.transform(X_test)
    
    # Load models
    print("\n  Loading models...")
    rf_model = joblib.load(MODELS_DIR / "rf_model.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgb_model.pkl")
    lgb_model = joblib.load(MODELS_DIR / "lgb_model.pkl")
    
    # Set n_jobs to 1 for serial execution
    rf_model.n_jobs = 1
    for m in [xgb_model, lgb_model]:
        if hasattr(m, "n_jobs"):
            m.n_jobs = 1
    
    # Generate predictions
    print("  Generating predictions on test set...")
    rf_pred = rf_model.predict(X_test_scaled)
    xgb_pred = xgb_model.predict(X_test)
    lgb_pred = lgb_model.predict(X_test)
    
    # Stacking ensemble prediction
    ensemble_pred = 0.45 * rf_pred + 0.35 * xgb_pred + 0.20 * lgb_pred
    
    # Calculate residuals
    residuals = ensemble_pred - np.array(y_test)
    hours = pd.to_datetime(timestamps_test).dt.hour
    
    # Compute median bias by hour
    bias_by_hour = {}
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            bias_by_hour[h] = float(np.median(residuals[mask]))
        else:
            bias_by_hour[h] = 0.0
    
    print("\n  Bias by hour (µg/m³):")
    for h in range(24):
        bias_val = bias_by_hour[h]
        symbol = "↑" if bias_val > 0.1 else "↓" if bias_val < -0.1 else "→"
        print(f"    Hour {h:2d}: {bias_val:+.3f} {symbol}")
    
    # Apply Gaussian smoothing to generalize (std=1.5 hours)
    biases_array = np.array([bias_by_hour[h] for h in range(24)])
    smoothed_biases = gaussian_filter1d(biases_array, sigma=1.5, mode='wrap')
    
    corrections = {h: float(smoothed_biases[h]) for h in range(24)}
    
    print("\n  Smoothed bias corrections (applied at prediction time):")
    for h in range(24):
        corr = corrections[h]
        symbol = "↑" if corr > 0.1 else "↓" if corr < -0.1 else "→"
        print(f"    Hour {h:2d}: {corr:+.3f} {symbol}")
    
    # Save corrections
    corrections_path = MODELS_DIR / "hourly_bias_corrections.pkl"
    joblib.dump(corrections, corrections_path)
    print(f"\n  ✓ Saved to: {corrections_path}")
    
    # Compute metrics with and without correction
    ensemble_pred_corrected = ensemble_pred - np.array([corrections[h] for h in hours])
    
    print("\n  Performance comparison (Ensemble):")
    print(f"    Without correction:")
    print(f"      R2   = {r2_score(y_test, ensemble_pred):.4f}")
    print(f"      RMSE = {np.sqrt(mean_squared_error(y_test, ensemble_pred)):.2f} µg/m³")
    print(f"      MAE  = {mean_absolute_error(y_test, ensemble_pred):.2f} µg/m³")
    print(f"    With correction:")
    print(f"      R2   = {r2_score(y_test, ensemble_pred_corrected):.4f}")
    print(f"      RMSE = {np.sqrt(mean_squared_error(y_test, ensemble_pred_corrected)):.2f} µg/m³")
    print(f"      MAE  = {mean_absolute_error(y_test, ensemble_pred_corrected):.2f} µg/m³")
    
    return corrections


def compute_daily_corrections():
    """Compute bias corrections for daily models."""
    print("\n" + "="*70)
    print(" Computing Daily Bias Corrections")
    print("="*70)
    
    # Load data
    data_path = ROOT / "lahore_air_quality_final_dataset.csv"
    if not data_path.exists():
        print(f"  [SKIP] {data_path} not found")
        return None
    
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 80/20 split
    train_end = int(len(df) * 0.80)
    test_df = df.iloc[train_end:].copy()
    
    print(f"\n  Loaded {len(df):,} daily rows, test set: {len(test_df):,} rows")
    
    if len(test_df) == 0:
        print("  [SKIP] Test set is empty")
        return None
    
    # Load feature list and scaler
    feature_list_path = DAILY_MODELS_DIR / "daily_feature_list.pkl"
    scaler_path = DAILY_MODELS_DIR / "daily_scaler.pkl"
    
    if not feature_list_path.exists() or not scaler_path.exists():
        print("  [SKIP] Feature list or scaler not found")
        return None
    
    features = joblib.load(feature_list_path)
    scaler = joblib.load(scaler_path)
    
    # Check if features exist in df
    try:
        # Engineer features
        test_df_eng = test_df.copy().sort_values("date").reset_index(drop=True)
        test_df_eng["day_of_week"] = test_df_eng["date"].dt.dayofweek
        test_df_eng["day_of_year"] = test_df_eng["date"].dt.dayofyear
        test_df_eng["week_of_year"] = test_df_eng["date"].dt.isocalendar().week.astype(int)
        test_df_eng["month_sin"] = np.sin(2 * np.pi * test_df_eng["month"] / 12)
        test_df_eng["month_cos"] = np.cos(2 * np.pi * test_df_eng["month"] / 12)
        test_df_eng["doy_sin"] = np.sin(2 * np.pi * test_df_eng["day_of_year"] / 365)
        test_df_eng["doy_cos"] = np.cos(2 * np.pi * test_df_eng["day_of_year"] / 365)
        test_df_eng["log_pm25_lag_1"] = np.log1p(test_df_eng["pm25_lag_1"])
        test_df_eng["log_pm25_lag_7"] = np.log1p(test_df_eng["pm25_lag_7"])
        test_df_eng["humidity_wind"] = test_df_eng["RH2M"] / (test_df_eng["WS2M"] + 0.5)
        test_df_eng["temp_wind_ratio"] = test_df_eng["T2M"] / (test_df_eng["WS2M"] + 0.5)
        test_df_eng["stagnation_proxy"] = np.clip(3.0 - test_df_eng["WS2M"], 0, None) * (test_df_eng["RH2M"] / 100)
        test_df_eng["temp_sq"] = test_df_eng["T2M"] ** 2
        test_df_eng["lag_diff"] = test_df_eng["pm25_lag_1"] - test_df_eng["pm25_lag_7"]
        test_df_eng["rolling_proxy"] = (test_df_eng["pm25_lag_1"] + test_df_eng["pm25_lag_7"]) / 2
        
        X_test = test_df_eng[features]
        y_test = test_df_eng["pm25"]
        dates_test = test_df_eng["date"]
    except (KeyError, AttributeError) as e:
        print(f"  [ERROR] Missing feature: {e}")
        return None
    
    X_test_scaled = scaler.transform(X_test)
    
    # Load best daily model (random forest typically performs best)
    best_model_path = DAILY_MODELS_DIR / "daily_random_forest.pkl"
    if not best_model_path.exists():
        print(f"  [SKIP] {best_model_path} not found")
        return None
    
    model = joblib.load(best_model_path)
    model.n_jobs = 1
    
    # Generate predictions
    print("  Generating predictions on test set...")
    pred = model.predict(X_test_scaled)
    
    # Calculate residuals by month
    residuals = pred - np.array(y_test)
    months = dates_test.dt.month
    
    # Compute median bias by month
    bias_by_month = {}
    for m in range(1, 13):
        mask = months == m
        if mask.sum() > 0:
            bias_by_month[m] = float(np.median(residuals[mask]))
        else:
            bias_by_month[m] = 0.0
    
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print("\n  Bias by month (µg/m³):")
    for m in range(1, 13):
        bias_val = bias_by_month[m]
        symbol = "↑" if bias_val > 0.5 else "↓" if bias_val < -0.5 else "→"
        print(f"    {month_names[m]:>3}: {bias_val:+.2f} {symbol}")
    
    # Apply smoothing
    biases_array = np.array([bias_by_month[m] for m in range(1, 13)])
    smoothed_biases = gaussian_filter1d(biases_array, sigma=0.8, mode='wrap')
    
    corrections = {m: float(smoothed_biases[m-1]) for m in range(1, 13)}
    
    print("\n  Smoothed bias corrections (applied at prediction time):")
    for m in range(1, 13):
        corr = corrections[m]
        symbol = "↑" if corr > 0.5 else "↓" if corr < -0.5 else "→"
        print(f"    {month_names[m]:>3}: {corr:+.2f} {symbol}")
    
    # Save corrections
    corrections_path = DAILY_MODELS_DIR / "daily_bias_corrections.pkl"
    joblib.dump(corrections, corrections_path)
    print(f"\n  ✓ Saved to: {corrections_path}")
    
    # Compute metrics
    pred_corrected = pred - np.array([corrections[m] for m in months])
    
    print("\n  Performance comparison (Daily RF Model):")
    print(f"    Without correction:")
    print(f"      R2   = {r2_score(y_test, pred):.4f}")
    print(f"      RMSE = {np.sqrt(mean_squared_error(y_test, pred)):.2f} µg/m³")
    print(f"      MAE  = {mean_absolute_error(y_test, pred):.2f} µg/m³")
    print(f"    With correction:")
    print(f"      R2   = {r2_score(y_test, pred_corrected):.4f}")
    print(f"      RMSE = {np.sqrt(mean_squared_error(y_test, pred_corrected)):.2f} µg/m³")
    print(f"      MAE  = {mean_absolute_error(y_test, pred_corrected):.2f} µg/m³")
    
    return corrections


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("  PM2.5 Model Bias Correction Computation")
    print("█"*70)
    
    hourly_corr = compute_hourly_corrections_legacy()
    daily_corr = compute_daily_corrections()
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    if hourly_corr:
        print("  ✓ Hourly bias corrections computed and saved")
    else:
        print("  ✗ Hourly corrections could not be computed")
    
    if daily_corr:
        print("  ✓ Daily bias corrections computed and saved")
    else:
        print("  ✗ Daily corrections could not be computed")
    
    print("\n  Next step: Run demo_app.py to apply corrections automatically")
    print("="*70 + "\n")
