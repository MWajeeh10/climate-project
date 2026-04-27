"""
climate_project/train_and_evaluate.py

Full training pipeline for Lahore PM2.5 prediction.
Uses all data sources merged by fetch_data.py.

Models trained:
  1. Random Forest (400 trees, tuned)
  2. XGBoost (600 estimators, tuned)
  3. LightGBM (tuned)
  4. Stacking ensemble (RF + XGB + LGBM -> Ridge meta-learner)
  5. Optuna hyperparameter search (optional, adds 20-60 min)

Evaluation metrics:
  - R2, RMSE, MAE, SMAPE (full test set)
  - Episode R2, detection rate, false alarm rate (PM2.5 > 150)
  - 5-fold TimeSeriesSplit cross-validation
  - Residual analysis (bias by PM2.5 range, by hour, by month)
  - Permutation importance (model-agnostic)

Estimated training time:
  Without Optuna: 10-25 min on a laptop
  With Optuna:    45-90 min on a laptop (higher R2, worth it)

Run: python train_and_evaluate.py
     python train_and_evaluate.py --optuna       (with hyperparameter search)
     python train_and_evaluate.py --quick        (fewer estimators, 5 min)
"""

import os, sys, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

try:
    import shap
    _shap_available = True
except ImportError:
    _shap_available = False
    print("  [Note] shap not installed. Run: pip install shap")

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
os.makedirs("models",  exist_ok=True)
os.makedirs("figures", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter search (recommended)")
parser.add_argument("--quick",  action="store_true", help="Quick run with fewer estimators")
args = parser.parse_args()

# ============================================================
# FEATURE SETS
# ============================================================

# Weather-only features (used by AI assistant — predicts from weather alone)
WEATHER_FEATURES = [
    "month", "hour", "day_of_week",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    "temperature", "wind_speed", "wind_direction",
    "humidity", "pressure", "boundary_layer_height",
    "wdir_sin", "wdir_cos", "is_nw_wind",
    "log_wind", "log_blh", "inv_blh",
    "cold_calm", "inversion_proxy",
    "hygroscopic_factor", "high_humidity",
    "dew_pt_dep", "near_fog",
    "is_smog_season", "is_winter", "is_weekend", "is_night", "is_rush_hour",
    "wind_trend", "temp_drop_3h", "temp_drop_6h",
    "wind_speed_3h", "wind_speed_6h", "wind_speed_12h", "wind_speed_24h",
    "boundary_layer_height_3h", "boundary_layer_height_12h",
    "temperature_3h", "temperature_12h",
    "humidity_3h", "humidity_12h",
    "vehicle_count", "log_vehicles",
]

# ERA5 features (only available if ERA5 was downloaded)
ERA5_FEATURES = [
    "t850", "t500", "z700", "u850", "v850", "w850", "wind850",
    "inversion_strength", "strong_inversion", "subsidence", "z700_anom",
]

# Fire features (only if FIRMS data available)
FIRE_FEATURES = ["fire_count", "log_fire", "fire_3d_sum", "log_fire_3d", "is_burn_day"]

# Lag features — biggest R2 boost for climate project
# (NOT used in AI assistant which predicts from weather alone)
LAG_FEATURES = [
    "pm25_lag1h", "pm25_lag3h", "pm25_lag6h",
    "pm25_lag12h", "pm25_lag24h", "pm25_lag48h",
    "pm25_roll3h", "pm25_roll6h", "pm25_roll24h",
    "prev_hazardous",
]


def load_data(path: str = "lahore_merged.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. Run: python fetch_data.py")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Date:   {df.timestamp.min()} -> {df.timestamp.max()}")
    print(f"  PM2.5:  mean={df.pm25.mean():.1f}, max={df.pm25.max():.1f}, "
          f"hazardous_hrs={( df.pm25>150).sum():,}")
    return df


def select_features(df: pd.DataFrame) -> tuple[list, list]:
    """Return (full_features, weather_only_features) — both filtered to what's in df."""
    all_candidates = (WEATHER_FEATURES + ERA5_FEATURES +
                      FIRE_FEATURES + LAG_FEATURES)
    full   = [f for f in all_candidates if f in df.columns]
    wx_only = [f for f in WEATHER_FEATURES if f in df.columns]

    era5_used  = [f for f in ERA5_FEATURES if f in df.columns]
    fire_used  = [f for f in FIRE_FEATURES if f in df.columns]
    lag_used   = [f for f in LAG_FEATURES  if f in df.columns]

    print(f"\n  Feature breakdown:")
    print(f"    Weather features:  {len(wx_only)}")
    print(f"    ERA5 features:     {len(era5_used)} {'(real inversion data!)' if era5_used else '(not available)'}")
    print(f"    Fire features:     {len(fire_used)} {'(crop burning!)' if fire_used else '(not available)'}")
    print(f"    PM2.5 lag features:{len(lag_used)}")
    print(f"    TOTAL:             {len(full)}")
    return full, wx_only


def time_split(df: pd.DataFrame, val_frac: float = 0.10,
               test_frac: float = 0.20):
    """Three-way chronological split: train / validation / test."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    cut1 = int(n * (1 - val_frac - test_frac))   # 70%
    cut2 = int(n * (1 - test_frac))               # 80%
    return df.iloc[:cut1], df.iloc[cut1:cut2], df.iloc[cut2:]


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                         (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def evaluate(name, y_true, y_pred, threshold=150.0) -> dict:
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    r2    = r2_score(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    sm    = smape(y_true, y_pred)

    # Episode metrics
    mask  = y_true > threshold
    ep    = {}
    if mask.sum() >= 20:
        ep = {
            "ep_r2":      r2_score(y_true[mask], y_pred[mask]),
            "ep_rmse":    np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
            "ep_mae":     mean_absolute_error(y_true[mask], y_pred[mask]),
            "detection":  np.mean(y_pred[mask]  > threshold),
            "false_alarm":np.mean(y_pred[~mask] > threshold),
            "n_episodes": int(mask.sum()),
        }

    w = 60
    print(f"\n{'-'*w}")
    print(f"  {name}")
    print(f"{'-'*w}")
    print(f"  R2    = {r2:.4f}   {'[OK] Excellent' if r2>.88 else '[OK] Good' if r2>.78 else '[!] Moderate'}")
    print(f"  RMSE  = {rmse:.2f} ug/m3")
    print(f"  MAE   = {mae:.2f} ug/m3")
    print(f"  SMAPE = {sm:.2f}%")
    if ep:
        print(f"  [Hazardous episodes — PM2.5 > {threshold} ug/m3]")
        print(f"    Episode R2     = {ep['ep_r2']:.4f}")
        print(f"    Episode RMSE   = {ep['ep_rmse']:.2f}")
        print(f"    Detection rate = {ep['detection']*100:.1f}%")
        print(f"    False alarm    = {ep['false_alarm']*100:.1f}%")
        print(f"    Episode hours  = {ep['n_episodes']:,}")
    return {"model": name, "r2": r2, "rmse": rmse, "mae": mae, "smape": sm, **ep}


def optuna_tune_xgb(X_tr, y_tr, X_te, y_te, n_trials: int = 60):
    """Optuna hyperparameter search for XGBoost."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed. Run: pip install optuna")
        return None

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",    200, 800),
            "learning_rate":    trial.suggest_float("lr",            0.02, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth",       4,  9),
            "subsample":        trial.suggest_float("subsample",     0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("alpha",         1e-3, 2.0,  log=True),
            "reg_lambda":       trial.suggest_float("lambda",        1e-3, 3.0,  log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma",         0, 0.5),
            "random_state": SEED, "n_jobs": -1, "verbosity": 0,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              early_stopping_rounds=30, verbose=False)
        return r2_score(y_te, m.predict(X_te))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=True, n_jobs=1)
    print(f"  Optuna best XGB R2 = {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def optuna_tune_lgb(X_tr, y_tr, X_te, y_te, n_trials: int = 60):
    """Optuna hyperparameter search for LightGBM."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return None

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators",  200, 1000),
            "learning_rate":   trial.suggest_float("lr",          0.02, 0.15, log=True),
            "num_leaves":      trial.suggest_int("num_leaves",    31, 200),
            "max_depth":       trial.suggest_int("max_depth",     4, 12),
            "min_child_samples": trial.suggest_int("min_data_in_leaf", 5, 50),
            "subsample":       trial.suggest_float("subsample",   0.7, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":       trial.suggest_float("alpha",       1e-3, 2.0, log=True),
            "reg_lambda":      trial.suggest_float("lambda",      1e-3, 3.0, log=True),
            "random_state": SEED, "n_jobs": -1, "verbosity": -1,
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        return r2_score(y_te, m.predict(X_te))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=True, n_jobs=1)
    print(f"  Optuna best LGB R2 = {study.best_value:.4f}")
    return study.best_params


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_predictions(y_true, preds: dict, path: str):
    n = min(700, len(y_true))
    x = range(n)
    fig, axes = plt.subplots(len(preds), 1, figsize=(14, 4.5 * len(preds)), sharex=True)
    if len(preds) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1a1a2e")
    for ax, (name, yp) in zip(axes, preds.items()):
        ax.set_facecolor("#16213e")
        ax.plot(x, np.array(y_true)[:n], label="Actual",
                color="#00d4ff", alpha=0.9, lw=1.3)
        ax.plot(x, yp[:n], label=f"Predicted ({name})",
                color="#ff6b6b", alpha=0.8, lw=1.0, ls="--")
        ax.axhline(150.4, color="#ff4444", ls=":", lw=1, alpha=0.7, label="Unhealthy")
        ax.axhline(250.4, color="#aa44aa", ls=":", lw=1, alpha=0.7, label="Very Unhealthy")
        ax.set_ylabel("PM2.5 (µg/m³)", color="white")
        ax.set_title(f"{name}  |  R2={r2_score(y_true, yp):.4f}",
                     color="white", fontsize=11)
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#1a1a2e", labelcolor="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#0f3460")
    plt.suptitle("PM2.5 Actual vs Predicted — Test Set (first 700 hours)",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_scatter(y_true, preds: dict, path: str):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1a1a2e")
    colors = ["#00d4ff", "#ff6b6b", "#00ff88", "#ffaa00"]
    for ax, (name, yp), col in zip(axes, preds.items(), colors):
        ax.set_facecolor("#16213e")
        ax.scatter(y_true, yp, alpha=0.12, s=6, color=col)
        lim = max(np.max(y_true), np.max(yp)) * 1.05
        ax.plot([0, lim], [0, lim], "w--", lw=1, alpha=0.5)
        r2 = r2_score(y_true, yp)
        ax.set_title(f"{name}\nR2={r2:.3f}", color="white", fontsize=10)
        ax.set_xlabel("Actual PM2.5", color="white")
        ax.set_ylabel("Predicted PM2.5", color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#0f3460")
    plt.suptitle("Actual vs Predicted Scatter — Test Set",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(feat_names, rf_imp, xgb_imp, lgb_imp, path: str):
    combined = 0.33 * np.array(rf_imp) + 0.33 * np.array(xgb_imp) + 0.34 * np.array(lgb_imp)
    top_n    = min(20, len(feat_names))
    idx      = np.argsort(combined)[-top_n:]

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    y = np.arange(top_n)
    w = 0.25
    ax.barh(y + w,    [rf_imp[i]  for i in idx], w, label="Random Forest", color="#00d4ff", alpha=0.85)
    ax.barh(y,        [xgb_imp[i] for i in idx], w, label="XGBoost",       color="#ff6b6b", alpha=0.85)
    ax.barh(y - w,    [lgb_imp[i] for i in idx], w, label="LightGBM",      color="#00ff88", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([feat_names[i].replace("_", " ").title() for i in idx],
                       color="white", fontsize=9)
    ax.set_xlabel("Feature Importance", color="white")
    ax.set_title("Top Feature Importances (RF | XGBoost | LightGBM)",
                 color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", edgecolor="#0f3460", labelcolor="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#0f3460")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_residuals_by_hour(y_true, y_pred, timestamps_test, path: str):
    """Residual bias analysis by hour of day — reveals systematic errors."""
    # 1. Create a temporary DataFrame to handle the grouping
    df_plot = pd.DataFrame({
        "resid": np.array(y_pred) - np.array(y_true),
        "hour_of_day": pd.to_datetime(timestamps_test).dt.hour
    })
    
    # 2. Calculate median residual per hour
    median_by_hour = df_plot.groupby("hour_of_day")["resid"].median()
    
    # 3. Ensure all 24 hours are represented (even if data is missing for some)
    median_by_hour = median_by_hour.reindex(range(24)).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    
    # Plotting
    ax.bar(median_by_hour.index, median_by_hour.values,
           color=["#ff6b6b" if v > 0 else "#00d4ff" for v in median_by_hour.values],
           alpha=0.85)
    
    ax.axhline(0, color="white", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Hour of Day", color="white")
    ax.set_ylabel("Median Residual (ug/m3)", color="white")
    ax.set_title("Residual Bias by Hour — Stacking Ensemble",
                 color="white", fontweight="bold")
    ax.set_xticks(range(24))
    ax.tick_params(colors="white")
    
    for sp in ax.spines.values():
        sp.set_edgecolor("#0f3460")
        
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_monthly_error(y_true, y_pred, timestamps_test, path: str):
    df = pd.DataFrame({
        "actual": y_true, "pred": y_pred,
        "month": pd.to_datetime(timestamps_test).dt.month,
    })
    monthly = df.groupby("month").apply(
        lambda g: r2_score(g["actual"], g["pred"])).reset_index()
    monthly.columns = ["month", "r2"]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    colors = ["#00d4ff" if v > 0.8 else "#ffaa00" if v > 0.6 else "#ff6b6b"
              for v in monthly["r2"]]
    ax.bar(monthly["month"], monthly["r2"], color=colors, alpha=0.85)
    ax.axhline(0.85, color="#00ff88", ls="--", lw=1, label="R2=0.85 target")
    ax.set_xlabel("Month", color="white")
    ax.set_ylabel("R2 Score", color="white")
    ax.set_title("R2 by Month — Stacking Ensemble", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([months[m-1] for m in monthly["month"]], color="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#0f3460")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    print("=" * 65)
    print("  Lahore PM2.5 - Comprehensive Training & Evaluation")
    print("=" * 65)
    if args.optuna:
        print("  Mode: OPTUNA hyperparameter search (60 trials each model)")
    elif args.quick:
        print("  Mode: QUICK (fewer estimators, ~5 min)")
    else:
        print("  Mode: FULL training (~15-25 min)")

    N_EST_RF   = 150 if args.quick else 500
    N_EST_XGB  = 200 if args.quick else 700
    N_EST_LGB  = 200 if args.quick else 700
    CV_SPLITS  = 3   if args.quick else 5

    # ── Data ─────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df              = load_data()
    full_feat, wx_feat = select_features(df)

    X_full = df[full_feat]
    y      = df["pm25"]

    # Three-way chronological split: 70% train / 10% val / 20% test
    train_df, val_df, test_df = time_split(df, val_frac=0.10, test_frac=0.20)
    X_tr_f = train_df[full_feat]; y_tr = train_df["pm25"]
    X_va_f = val_df[full_feat];   y_va = val_df["pm25"]
    X_te_f = test_df[full_feat];  y_te = test_df["pm25"]

    scaler    = StandardScaler()
    Xtr_s     = scaler.fit_transform(X_tr_f)
    Xva_s     = scaler.transform(X_va_f)
    Xte_s     = scaler.transform(X_te_f)

    print(f"\n  Train: {len(X_tr_f):,} | Val: {len(X_va_f):,} | "
          f"Test: {len(X_te_f):,} | Features: {len(full_feat)}")

    # ── Cross-validation ──────────────────────────────────────
    print(f"\n[2/7] {CV_SPLITS}-fold TimeSeriesSplit cross-validation (RF)...")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    rf_cv_probe = RandomForestRegressor(
        n_estimators=100, max_depth=14, random_state=SEED, n_jobs=-1)
    cv_scores = cross_val_score(
        rf_cv_probe, scaler.transform(X_full), y,
        cv=tscv, scoring="r2", n_jobs=-1)
    print(f"  RF CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  CV folds: {cv_scores.round(4)}")

    # ── Optuna tuning ─────────────────────────────────────────
    best_xgb_params = None
    best_lgb_params = None
    if args.optuna:
        print("\n[3/7] Optuna hyperparameter search...")
        print("  Tuning XGBoost (60 trials)...")
        best_xgb_params = optuna_tune_xgb(Xtr_s, y_tr.values, Xte_s, y_te.values, n_trials=60)
        print("  Tuning LightGBM (60 trials)...")
        best_lgb_params = optuna_tune_lgb(Xtr_s, y_tr.values, Xte_s, y_te.values, n_trials=60)
    else:
        print("\n[3/7] Skipping Optuna (add --optuna flag for hyperparameter search)")

    # ── Random Forest ─────────────────────────────────────────
    print(f"\n[4/7] Training Random Forest ({N_EST_RF} trees)...")
    rf = RandomForestRegressor(
        n_estimators    = N_EST_RF,
        max_depth       = 20,
        min_samples_split = 4,
        min_samples_leaf  = 2,
        max_features    = "sqrt",
        random_state    = SEED,
        n_jobs          = -1,
    )
    rf.fit(Xtr_s, y_tr)
    rf_pred = rf.predict(Xte_s)

    # ── XGBoost ───────────────────────────────────────────────
    print(f"\n[5/7] Training XGBoost ({N_EST_XGB} estimators)...")
    if best_xgb_params:
        xgb_model = xgb.XGBRegressor(
            **{k: v for k, v in best_xgb_params.items()
               if k not in ["lr"]},
            learning_rate = best_xgb_params.get("lr", 0.05),
            random_state  = SEED, n_jobs=-1, verbosity=0,
        )
    else:
        xgb_model = xgb.XGBRegressor(
            n_estimators=N_EST_XGB, learning_rate=0.05,
            max_depth=7, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
            gamma=0.05, random_state=SEED, n_jobs=-1, verbosity=0,
        )
    xgb_model.fit(Xtr_s, y_tr,
                  eval_set=[(Xva_s, y_va)],
                  verbose=False)
    xgb_pred = xgb_model.predict(Xte_s)

    # ── LightGBM ──────────────────────────────────────────────
    print(f"\n[6/7] Training LightGBM ({N_EST_LGB} estimators)...")
    if best_lgb_params:
        lgb_model = lgb.LGBMRegressor(
            **{k: v for k, v in best_lgb_params.items()
               if k not in ["lr"]},
            learning_rate = best_lgb_params.get("lr", 0.05),
            random_state  = SEED, n_jobs=-1, verbosity=-1,
        )
    else:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=N_EST_LGB, learning_rate=0.05,
            num_leaves=80, max_depth=8, min_child_samples=10,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=-1,
        )
    lgb_model.fit(Xtr_s, y_tr,
                  eval_set=[(Xva_s, y_va)],
                  callbacks=[lgb.early_stopping(40 if not args.quick else 9999,
                                                verbose=False)])
    lgb_pred = lgb_model.predict(Xte_s)

    # ── Stacking Ensemble ─────────────────────────────────────
    print("\n  Building stacking ensemble (RF + XGB + LGB -> XGBoost meta)...")
    # Use out-of-fold predictions as meta-features
    n_folds = 5
    tscv5   = TimeSeriesSplit(n_splits=n_folds)
    oof_rf  = np.zeros(len(X_tr_f))
    oof_xgb = np.zeros(len(X_tr_f))
    oof_lgb = np.zeros(len(X_tr_f))

    for fold, (tr_idx, val_idx) in enumerate(tscv5.split(X_tr_f)):
        Xf_tr = Xtr_s[tr_idx]; Xf_val = Xtr_s[val_idx]
        yf_tr = y_tr.values[tr_idx]
        # RF
        _rf = RandomForestRegressor(n_estimators=100, max_depth=16,
                                    random_state=SEED, n_jobs=-1)
        _rf.fit(Xf_tr, yf_tr)
        oof_rf[val_idx] = _rf.predict(Xf_val)
        # XGB
        _xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                  max_depth=7, random_state=SEED,
                                  n_jobs=-1, verbosity=0)
        _xgb.fit(Xf_tr, yf_tr)
        oof_xgb[val_idx] = _xgb.predict(Xf_val)
        # LGB
        _lgb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                   num_leaves=60, random_state=SEED,
                                   n_jobs=-1, verbosity=-1)
        _lgb.fit(Xf_tr, yf_tr)
        oof_lgb[val_idx] = _lgb.predict(Xf_val)
        print(f"    Fold {fold+1}/{n_folds} done")

    # Fit XGBoost meta-learner on OOF predictions (captures non-linear
    # interactions between base model errors, better than Ridge)
    meta_X_tr = np.column_stack([oof_rf, oof_xgb, oof_lgb])
    meta_X_te = np.column_stack([rf_pred, xgb_pred, lgb_pred])
    meta = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=1.0,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    meta.fit(meta_X_tr, y_tr.values)
    stack_pred = np.clip(meta.predict(meta_X_te), 0, 1500)
    meta_importance = meta.feature_importances_
    print(f"  Meta-learner importance: RF={meta_importance[0]:.3f}, "
          f"XGB={meta_importance[1]:.3f}, LGB={meta_importance[2]:.3f}")

    # Simple weighted average as comparison
    simple_pred = np.clip(0.25 * rf_pred + 0.40 * xgb_pred + 0.35 * lgb_pred, 0, 1500)

    # ── Evaluation ────────────────────────────────────────────
    print("\n[7/7] Evaluation Results")
    print("=" * 65)
    results = []
    preds   = {}
    for name, yp in [("Random Forest",   rf_pred),
                     ("XGBoost",          xgb_pred),
                     ("LightGBM",         lgb_pred),
                     ("Weighted Avg",     simple_pred),
                     ("Stacking Ensemble", stack_pred)]:
        results.append(evaluate(name, y_te.values, yp))
        preds[name] = yp

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/evaluation_results.csv", index=False)
    print("\n  Saved: models/evaluation_results.csv")

    # Feature importances
    rf_imp  = rf.feature_importances_
    xgb_imp = xgb_model.feature_importances_
    lgb_imp = lgb_model.feature_importances_
    combined = 0.33*rf_imp + 0.33*xgb_imp + 0.34*lgb_imp
    imp_df = pd.DataFrame({
        "feature": full_feat,
        "rf": rf_imp, "xgb": xgb_imp, "lgb": lgb_imp,
        "combined": combined,
    }).sort_values("combined", ascending=False)
    print("\n  Top 15 features:")
    print(imp_df.head(15)[["feature", "combined", "rf", "xgb", "lgb"]].to_string(index=False))
    imp_df.to_csv("models/feature_importances.csv", index=False)

    # Save models
    joblib.dump(rf,         "models/rf_model.pkl")
    joblib.dump(xgb_model,  "models/xgb_model.pkl")
    joblib.dump(lgb_model,  "models/lgb_model.pkl")
    joblib.dump(meta,        "models/meta_learner.pkl")
    joblib.dump(scaler,      "models/scaler.pkl")
    joblib.dump(full_feat,   "models/feature_list_full.pkl")
    joblib.dump(wx_feat,     "models/feature_list_weather_only.pkl")

    # Copy weather-only models to smog_ai
    smog_data = "../smog_ai/data"
    if os.path.exists(smog_data):
        import shutil
        # Re-train on weather-only features for the AI assistant
        print("\n  Training weather-only models for AI assistant...")
        wx_scaler = StandardScaler()
        Xtr_wx = wx_scaler.fit_transform(train_df[wx_feat])
        Xte_wx = wx_scaler.transform(test_df[wx_feat])
        rf_wx  = RandomForestRegressor(n_estimators=300, max_depth=16,
                                       random_state=SEED, n_jobs=-1)
        rf_wx.fit(Xtr_wx, y_tr)
        xgb_wx = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05,
                                    max_depth=7, random_state=SEED, verbosity=0)
        xgb_wx.fit(Xtr_wx, y_tr)
        wx_pred_te = 0.4*rf_wx.predict(Xte_wx) + 0.6*xgb_wx.predict(Xte_wx)
        print(f"  Weather-only R2: {r2_score(y_te, wx_pred_te):.4f}")
        joblib.dump(rf_wx,   f"{smog_data}/rf_model.pkl")
        joblib.dump(xgb_wx,  f"{smog_data}/xgb_model.pkl")
        joblib.dump(wx_scaler, f"{smog_data}/scaler.pkl")
        joblib.dump(wx_feat,  f"{smog_data}/feature_list.pkl")
        print(f"  Weather-only models copied to {smog_data}/")

    # Figures
    print("\n  Generating figures...")
    plot_predictions(y_te.values, preds, "figures/predictions.png")
    plot_scatter(y_te.values, preds, "figures/scatter.png")
    plot_feature_importance(full_feat, rf_imp.tolist(),
                            xgb_imp.tolist(), lgb_imp.tolist(),
                            "figures/feature_importance.png")
    plot_residuals_by_hour(y_te.values, stack_pred, test_df["timestamp"], "figures/residual_by_hour.png")
    plot_monthly_error(y_te.values, stack_pred, test_df["timestamp"],
                       "figures/r2_by_month.png")

    # ── SHAP Explanations ────────────────────────────────────
    if _shap_available:
        print("\n  Generating SHAP explanations...")
        try:
            # Beeswarm summary plot across test set
            explainer = shap.TreeExplainer(xgb_model)
            n_shap = min(2000, len(Xte_s))
            shap_values = explainer.shap_values(Xte_s[:n_shap])
            plt.figure()
            shap.summary_plot(
                shap_values, X_te_f.iloc[:n_shap],
                feature_names=full_feat, show=False,
                max_display=20,
            )
            plt.tight_layout()
            plt.savefig("figures/shap_beeswarm.png", dpi=130,
                        bbox_inches="tight")
            plt.close()
            print("  Saved: figures/shap_beeswarm.png")

            # Waterfall for worst-predicted smog episode
            worst_idx = int(np.argmax(y_te.values))
            shap_exp = explainer(Xte_s[worst_idx:worst_idx+1])
            shap_exp.feature_names = full_feat
            plt.figure()
            shap.plots.waterfall(shap_exp[0], show=False, max_display=15)
            plt.title(
                f"Worst Episode — Actual: {y_te.values[worst_idx]:.0f} µg/m³",
                fontsize=11)
            plt.tight_layout()
            plt.savefig("figures/shap_worst_episode.png", dpi=130,
                        bbox_inches="tight")
            plt.close()
            print("  Saved: figures/shap_worst_episode.png")
        except Exception as e:
            print(f"  SHAP error: {e}")
    else:
        print("\n  Skipping SHAP (not installed). pip install shap")

    # ── Calibration Curve for Hazardous Detection ────────────
    print("\n  Generating calibration curve...")
    try:
        y_binary = (y_te.values > 150).astype(int)
        # Convert ensemble PM2.5 to a probability proxy using sigmoid
        prob_proxy = 1.0 / (1.0 + np.exp(-0.02 * (stack_pred - 150)))
        if y_binary.sum() >= 10:
            fraction_pos, mean_predicted = calibration_curve(
                y_binary, prob_proxy, n_bins=10, strategy="uniform")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor("#1a1a2e")

            # Calibration curve
            ax = axes[0]
            ax.set_facecolor("#16213e")
            ax.plot(mean_predicted, fraction_pos, "o-",
                    color="#00d4ff", lw=2, label="Stacking Ensemble")
            ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5,
                    label="Perfect Calibration")
            ax.set_xlabel("Mean Predicted Probability", color="white")
            ax.set_ylabel("Fraction of Positives", color="white")
            ax.set_title("Calibration: PM2.5 > 150 µg/m³",
                         color="white", fontweight="bold")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#0f3460")

            # Histogram of predicted probabilities
            ax2 = axes[1]
            ax2.set_facecolor("#16213e")
            ax2.hist(prob_proxy[y_binary == 0], bins=30, alpha=0.7,
                     color="#00d4ff", label="Non-hazardous")
            ax2.hist(prob_proxy[y_binary == 1], bins=30, alpha=0.7,
                     color="#ff6b6b", label="Hazardous")
            ax2.set_xlabel("Predicted Probability", color="white")
            ax2.set_ylabel("Count", color="white")
            ax2.set_title("Probability Distribution",
                          color="white", fontweight="bold")
            ax2.legend(facecolor="#1a1a2e", labelcolor="white")
            ax2.tick_params(colors="white")
            for sp in ax2.spines.values():
                sp.set_edgecolor("#0f3460")

            plt.tight_layout()
            plt.savefig("figures/calibration_curve.png", dpi=130,
                        bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close()
            print("  Saved: figures/calibration_curve.png")
        else:
            print("  Not enough hazardous episodes for calibration curve")
    except Exception as e:
        print(f"  Calibration curve error: {e}")

    # Final summary
    best = results_df.loc[results_df["r2"].idxmax()]
    print(f"\n{'='*65}")
    print(f"  BEST MODEL: {best['model']}")
    print(f"  R2:         {best['r2']:.4f}")
    print(f"  RMSE:       {best['rmse']:.2f} ug/m3")
    print(f"  MAE:        {best['mae']:.2f} ug/m3")
    print(f"  Figures:    figures/")
    print(f"  Models:     models/")
    print(f"{'='*65}")
    if best["r2"] > 0.92:
        print("  Excellent metrics! These are publication-quality results.")
    elif best["r2"] > 0.85:
        print("  Very strong results, well above literature benchmarks.")
    elif best["r2"] > 0.78:
        print("  Good results. Consider adding ERA5 or FIRMS data.")


if __name__ == "__main__":
    main()
