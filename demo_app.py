import csv
import json
import math
import mimetypes
import warnings
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
DAILY_MODELS_DIR = ROOT / "models_observed_daily"
RAW_DATA_DIR = ROOT / "raw_data"

HOST = "127.0.0.1"
PORT = 8080


def _float(value, default):
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values, default):
    clean = [float(v) for v in values if v is not None]
    return float(sum(clean) / len(clean)) if clean else float(default)


def load_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class DemoArtifacts:
    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "rf_model.pkl")
        self.model.n_jobs = 1
        self.xgb_model = joblib.load(MODELS_DIR / "xgb_model.pkl")
        self.lgb_model = joblib.load(MODELS_DIR / "lgb_model.pkl")
        self.meta_model = joblib.load(MODELS_DIR / "meta_learner.pkl")
        for model in (self.xgb_model, self.lgb_model, self.meta_model):
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.features = joblib.load(MODELS_DIR / "feature_list_full.pkl")
        
        # Load bias corrections
        self.hourly_bias_corrections = self._load_bias_corrections()
        
        self.vehicle_lookup = self._load_vehicle_lookup()
        self.evaluation_results = self._load_evaluation_results()
        self.best_saved_model = max(self.evaluation_results, key=lambda item: item["r2"])["model"]
        self.top_features = self._load_top_features()
        self.figure_cards = self._load_figure_cards()
        self.presets = self._build_presets()
        self.research_highlights = self._build_research_highlights()
        self.glossary = self._build_glossary()
        self.calculation_steps = self._build_calculation_steps()
        self.climatology = self._load_climatology()
        # Load historical hourly data for accurate past-timestamp queries (Improvement A)
        self.history_hourly = self._load_history_hourly()

    def _load_history_hourly(self):
        """Load historical hourly dataset, indexed by rounded timestamp for O(1) lookups."""
        path = ROOT / "lahore_merged.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path, parse_dates=["timestamp"])
        # Round timestamps to the nearest hour and build a dict keyed by ISO string
        df["timestamp"] = df["timestamp"].dt.floor("h")
        lookup = {}
        for _, row in df.iterrows():
            key = row["timestamp"].strftime("%Y-%m-%dT%H:00")
            lookup[key] = row.to_dict()
        print(f"  [History] Loaded {len(lookup):,} hourly rows for historical lookups")
        return lookup

    def _load_bias_corrections(self):
        """Load hour-specific bias corrections for PM2.5 predictions."""
        corrections_path = MODELS_DIR / "hourly_bias_corrections.pkl"
        if corrections_path.exists():
            try:
                return joblib.load(corrections_path)
            except Exception as e:
                print(f"  [Warning] Could not load bias corrections: {e}")
        # Return zero corrections if file doesn't exist
        return {h: 0.0 for h in range(24)}

    def _load_vehicle_lookup(self):
        rows = load_csv_rows(RAW_DATA_DIR / "vehicle_data.csv")
        lookup = {}
        for row in rows:
            try:
                lookup[int(row["year"])] = float(row["vehicle_count"])
            except (KeyError, TypeError, ValueError):
                continue
        return lookup or {2023: 6_290_000.0}

    def _vehicle_count_for_year(self, year):
        if year in self.vehicle_lookup:
            return self.vehicle_lookup[year]

        years = sorted(self.vehicle_lookup)
        first_year, last_year = years[0], years[-1]
        first_val = self.vehicle_lookup[first_year]
        last_val = self.vehicle_lookup[last_year]
        slope = (last_val - first_val) / max(1, last_year - first_year)

        if year < first_year:
            return first_val + slope * (year - first_year)
        return last_val + slope * (year - last_year)

    def _load_evaluation_results(self):
        rows = load_csv_rows(MODELS_DIR / "evaluation_results.csv")
        cleaned = []
        for row in rows:
            try:
                cleaned.append({
                    "model": row["model"],
                    "r2": round(float(row["r2"]), 4),
                    "rmse": round(float(row["rmse"]), 2),
                    "mae": round(float(row["mae"]), 2),
                    "smape": round(float(row["smape"]), 2),
                    "episode_r2": round(float(row["ep_r2"]), 4),
                    "episode_rmse": round(float(row["ep_rmse"]), 2),
                    "hazard_detection": round(float(row["detection"]) * 100, 1),
                    "false_alarm": round(float(row["false_alarm"]) * 100, 1),
                    "episode_hours": int(float(row["n_episodes"])),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return cleaned

    def _load_top_features(self):
        rows = load_csv_rows(MODELS_DIR / "feature_importances.csv")
        parsed = []
        for row in rows:
            try:
                parsed.append({
                    "feature": row["feature"],
                    "combined": float(row["combined"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
        parsed.sort(key=lambda item: item["combined"], reverse=True)
        return parsed[:10]

    def _load_figure_cards(self):
        cards = [
            {
                "title": "Prediction Traces",
                "caption": "How closely the saved models follow actual smog spikes on the held-out test window.",
                "reading": "The near-overlap between actual and predicted curves shows that the trained system captures both severe surges and sudden collapses.",
                "impact": "Useful in the demo because it immediately shows that the model is not just predicting an average line.",
                "path": "/figures/predictions.png",
            },
            {
                "title": "Model Fit Scatter",
                "caption": "Predicted versus actual PM2.5 values for the trained models.",
                "reading": "Tight clustering around the diagonal indicates strong fit. Wider spread would imply unstable prediction behavior.",
                "impact": "Helpful for non-technical audiences because it visually communicates accuracy without jargon.",
                "path": "/figures/scatter.png",
            },
            {
                "title": "Feature Story",
                "caption": "Top features learned by the training pipeline.",
                "reading": "Boundary layer height, lagged PM2.5, wind speed, and time-of-day effects dominate the model signal.",
                "impact": "This graph supports the argument that the system is learning physical smog behavior, not random correlations.",
                "path": "/figures/feature_importance.png",
            },
            {
                "title": "Explainability",
                "caption": "SHAP summary view of what pushes smog risk upward or downward.",
                "reading": "It shows which variables tend to push the prediction into cleaner or dirtier regimes across many examples.",
                "impact": "Important if you want to claim explainability and policy relevance in a paper or jury demo.",
                "path": "/figures/shap_beeswarm.png",
            },
            {
                "title": "Calibration",
                "caption": "How well hazardous-episode risk aligns with observed outcomes.",
                "reading": "This checks whether a high risk score actually corresponds to real hazardous episodes instead of over-warning.",
                "impact": "Strong calibration matters if the project is presented as an early-warning or public decision-support tool.",
                "path": "/figures/calibration_curve.png",
            },
        ]
        return [card for card in cards if (ROOT / card["path"].lstrip("/")).exists()]

    def _build_research_highlights(self):
        return [
            {
                "title": "City-specific atmospheric framing",
                "body": "The project is tailored to Lahore's winter smog dynamics rather than using a generic air-quality template. It emphasizes boundary-layer trapping, seasonal stagnation, transport direction, and persistence effects.",
            },
            {
                "title": "Hybrid physics + machine learning story",
                "body": "Instead of only fitting a black-box model, the feature design reflects physical smog mechanisms such as shallow mixing depth, calm cold mornings, and likely smoke transport pathways.",
            },
            {
                "title": "Decision-support potential",
                "body": "Because the output can be turned into risk bands, plain-language narratives, and visual evidence, the work can grow from a prediction model into a public-facing warning and planning tool.",
            },
        ]

    def _build_glossary(self):
        return [
            {
                "term": "Boundary Layer Height",
                "meaning": "The vertical depth of air that can mix pollutants upward. Lower values usually mean dirtier air because emissions stay trapped near the ground.",
            },
            {
                "term": "Pollution Memory",
                "meaning": "Recent PM2.5 history. If the city was already polluted a few hours ago, conditions often remain bad unless weather clears the air.",
            },
            {
                "term": "Northwesterly Transport",
                "meaning": "Wind coming from the northwest can align with regional smoke transport patterns during the smog season.",
            },
            {
                "term": "Humidity / Near-Fog Signal",
                "meaning": "Moist air helps haze persist and often appears alongside stagnant winter smog episodes.",
            },
            {
                "term": "Risk Band",
                "meaning": "A simplified interpretation of PM2.5 severity so non-technical users can quickly understand whether conditions are clear, elevated, unhealthy, or extreme.",
            },
        ]

    def _build_calculation_steps(self):
        return [
            {
                "step": "1. Build the weather and smog context",
                "detail": "The app starts with the selected date and time, then uses either your manual inputs or typical Lahore historical conditions for that month and hour.",
            },
            {
                "step": "2. Engineer smog-sensitive features",
                "detail": "It converts raw inputs into model features such as humidity effects, boundary-layer trapping, wind direction signals, time-of-day cycles, and recent pollution memory.",
            },
            {
                "step": "3. Standardize the feature vector",
                "detail": "The saved scaler transforms the inputs into the same numerical space used when the models were trained.",
            },
            {
                "step": "4. Run multiple trained models",
                "detail": "Random Forest, XGBoost, and LightGBM each make a PM2.5 estimate from the same engineered feature set.",
            },
            {
                "step": "5. Blend the model outputs",
                "detail": "The headline live prediction uses the saved weighted ensemble because it had the strongest held-out benchmark in the project artifacts.",
            },
            {
                "step": "6. Translate the result for people",
                "detail": "The raw PM2.5 output is turned into a risk band, confidence interval, driver summary, and plain-language public guidance.",
            },
        ]

    def _load_climatology(self):
        path = ROOT / "lahore_merged.csv"
        fields = [
            "temperature", "dew_point", "humidity", "pressure", "wind_speed",
            "wind_direction", "boundary_layer_height", "wind_speed_3h",
            "wind_speed_6h", "wind_speed_12h", "wind_speed_24h",
            "boundary_layer_height_3h", "boundary_layer_height_12h",
            "temperature_3h", "temperature_12h", "humidity_3h", "humidity_12h",
            "pm25_lag1h", "pm25_lag3h", "pm25_lag6h", "pm25_lag12h",
            "pm25_lag24h", "pm25_lag48h",
        ]
        if not path.exists():
            return {"month_hour": {}, "month": {}}

        df = pd.read_csv(path, parse_dates=["timestamp"], usecols=["timestamp", *fields])
        df["month"] = df["timestamp"].dt.month
        df["hour"] = df["timestamp"].dt.hour

        month_hour = (
            df.groupby(["month", "hour"])[fields]
            .mean(numeric_only=True)
            .round(3)
            .to_dict(orient="index")
        )
        month = (
            df.groupby("month")[fields]
            .mean(numeric_only=True)
            .round(3)
            .to_dict(orient="index")
        )
        return {"month_hour": month_hour, "month": month}

    def auto_payload_for_timestamp(self, timestamp, traffic_factor=1.0):
        dt = datetime.fromisoformat(timestamp)
        key = dt.strftime("%Y-%m-%dT%H:00")
        now = datetime.now()
        is_past = dt < now

        # --- Improvement A: For past timestamps, look up actual recorded data ---
        if is_past and key in self.history_hourly:
            hist_row = self.history_hourly[key]
            payload = {"timestamp": timestamp, "traffic_factor": traffic_factor, "_source": "observed_history"}
            # Map all feature columns directly from the historical row
            for col in [
                "temperature", "dew_point", "humidity", "pressure", "wind_speed",
                "wind_direction", "boundary_layer_height",
                "wind_speed_3h", "wind_speed_6h", "wind_speed_12h", "wind_speed_24h",
                "boundary_layer_height_3h", "boundary_layer_height_12h",
                "temperature_3h", "temperature_12h", "humidity_3h", "humidity_12h",
                "pm25_lag1h", "pm25_lag3h", "pm25_lag6h",
                "pm25_lag12h", "pm25_lag24h", "pm25_lag48h",
            ]:
                if col in hist_row and hist_row[col] is not None and not (isinstance(hist_row[col], float) and math.isnan(hist_row[col])):
                    payload[col] = float(hist_row[col])
            payload.setdefault("dew_point", payload.get("temperature", 18.0) - 3.0)
            return payload

        # --- Fallback: use climatological averages (future or unknown dates) ---
        row = self.climatology["month_hour"].get((dt.month, dt.hour)) or self.climatology["month"].get(dt.month) or {}
        payload = {"timestamp": timestamp, "traffic_factor": traffic_factor, "_source": "climatology"}
        for key, value in row.items():
            payload[key] = float(value)

        payload.setdefault("temperature", 18.0)
        payload.setdefault("dew_point", payload["temperature"] - 3.0)
        payload.setdefault("humidity", 65.0)
        payload.setdefault("pressure", 1012.0)
        payload.setdefault("wind_speed", 2.4)
        payload.setdefault("wind_direction", 300.0)
        payload.setdefault("boundary_layer_height", 260.0)
        payload.setdefault("wind_speed_3h", payload["wind_speed"])
        payload.setdefault("wind_speed_6h", payload["wind_speed"])
        payload.setdefault("wind_speed_12h", payload["wind_speed"])
        payload.setdefault("wind_speed_24h", payload["wind_speed"])
        payload.setdefault("boundary_layer_height_3h", payload["boundary_layer_height"])
        payload.setdefault("boundary_layer_height_12h", payload["boundary_layer_height"])
        payload.setdefault("temperature_3h", payload["temperature"])
        payload.setdefault("temperature_12h", payload["temperature"])
        payload.setdefault("humidity_3h", payload["humidity"])
        payload.setdefault("humidity_12h", payload["humidity"])
        payload.setdefault("pm25_lag1h", 82.0)
        payload.setdefault("pm25_lag3h", payload["pm25_lag1h"])
        payload.setdefault("pm25_lag6h", payload["pm25_lag3h"])
        payload.setdefault("pm25_lag12h", payload["pm25_lag6h"])
        payload.setdefault("pm25_lag24h", payload["pm25_lag12h"])
        payload.setdefault("pm25_lag48h", payload["pm25_lag24h"])
        payload["temp_3h_ago"] = payload["temperature_3h"]
        payload["temp_6h_ago"] = payload["temperature_12h"]
        return payload

    def _build_presets(self):
        return [
            {
                "id": "winter-inversion-dawn",
                "title": "Winter Inversion Dawn",
                "description": "Cold, humid, stagnant dawn with a shallow mixing layer and elevated carryover pollution.",
                "payload": {
                    "timestamp": "2026-01-14T06:00",
                    "temperature": 8,
                    "dew_point": 6.8,
                    "humidity": 93,
                    "pressure": 1018,
                    "wind_speed": 1.1,
                    "wind_direction": 322,
                    "boundary_layer_height": 95,
                    "traffic_factor": 1.12,
                    "wind_speed_3h": 1.3,
                    "wind_speed_6h": 1.6,
                    "wind_speed_12h": 1.8,
                    "wind_speed_24h": 2.0,
                    "boundary_layer_height_3h": 105,
                    "boundary_layer_height_12h": 180,
                    "temperature_3h": 9.5,
                    "temperature_12h": 14.0,
                    "temp_3h_ago": 11,
                    "temp_6h_ago": 14,
                    "humidity_3h": 90,
                    "humidity_12h": 81,
                    "pm25_lag1h": 238,
                    "pm25_lag3h": 226,
                    "pm25_lag6h": 205,
                    "pm25_lag12h": 180,
                    "pm25_lag24h": 192,
                    "pm25_lag48h": 168,
                },
            },
            {
                "id": "crop-burning-surge",
                "title": "Crop Burning Surge",
                "description": "Post-monsoon smoke transport with northwesterly flow, weak wind, and already-elevated PM2.5.",
                "payload": {
                    "timestamp": "2026-11-03T08:00",
                    "temperature": 19,
                    "dew_point": 14.5,
                    "humidity": 76,
                    "pressure": 1013,
                    "wind_speed": 1.7,
                    "wind_direction": 305,
                    "boundary_layer_height": 160,
                    "traffic_factor": 1.1,
                    "wind_speed_3h": 1.9,
                    "wind_speed_6h": 2.1,
                    "wind_speed_12h": 2.5,
                    "wind_speed_24h": 2.7,
                    "boundary_layer_height_3h": 150,
                    "boundary_layer_height_12h": 220,
                    "temperature_3h": 18,
                    "temperature_12h": 22,
                    "temp_3h_ago": 17,
                    "temp_6h_ago": 16,
                    "humidity_3h": 78,
                    "humidity_12h": 72,
                    "pm25_lag1h": 188,
                    "pm25_lag3h": 176,
                    "pm25_lag6h": 162,
                    "pm25_lag12h": 148,
                    "pm25_lag24h": 156,
                    "pm25_lag48h": 131,
                },
            },
            {
                "id": "commute-trap-evening",
                "title": "Commute Trap Evening",
                "description": "Rush-hour emissions stack into a lowering boundary layer after sunset.",
                "payload": {
                    "timestamp": "2026-12-09T19:00",
                    "temperature": 14,
                    "dew_point": 10.8,
                    "humidity": 82,
                    "pressure": 1017,
                    "wind_speed": 1.5,
                    "wind_direction": 340,
                    "boundary_layer_height": 130,
                    "traffic_factor": 1.22,
                    "wind_speed_3h": 1.8,
                    "wind_speed_6h": 2.3,
                    "wind_speed_12h": 2.7,
                    "wind_speed_24h": 3.0,
                    "boundary_layer_height_3h": 175,
                    "boundary_layer_height_12h": 310,
                    "temperature_3h": 16,
                    "temperature_12h": 19,
                    "temp_3h_ago": 17,
                    "temp_6h_ago": 19,
                    "humidity_3h": 79,
                    "humidity_12h": 68,
                    "pm25_lag1h": 152,
                    "pm25_lag3h": 141,
                    "pm25_lag6h": 129,
                    "pm25_lag12h": 112,
                    "pm25_lag24h": 138,
                    "pm25_lag48h": 118,
                },
            },
            {
                "id": "rain-cleared-afternoon",
                "title": "Rain-Cleared Afternoon",
                "description": "Ventilated, washed-out air after rain with a higher mixing layer and weaker pollution memory.",
                "payload": {
                    "timestamp": "2026-07-22T15:00",
                    "temperature": 31,
                    "dew_point": 22,
                    "humidity": 58,
                    "pressure": 1002,
                    "wind_speed": 4.8,
                    "wind_direction": 210,
                    "boundary_layer_height": 820,
                    "traffic_factor": 0.95,
                    "wind_speed_3h": 4.5,
                    "wind_speed_6h": 4.2,
                    "wind_speed_12h": 3.9,
                    "wind_speed_24h": 3.6,
                    "boundary_layer_height_3h": 780,
                    "boundary_layer_height_12h": 620,
                    "temperature_3h": 30,
                    "temperature_12h": 28,
                    "temp_3h_ago": 29,
                    "temp_6h_ago": 27,
                    "humidity_3h": 61,
                    "humidity_12h": 68,
                    "pm25_lag1h": 32,
                    "pm25_lag3h": 36,
                    "pm25_lag6h": 41,
                    "pm25_lag12h": 45,
                    "pm25_lag24h": 49,
                    "pm25_lag48h": 55,
                },
            },
        ]

    def build_feature_vector(self, payload):
        timestamp = payload.get("timestamp") or datetime.now().strftime("%Y-%m-%dT%H:00")
        dt = datetime.fromisoformat(timestamp)
        traffic_factor = _float(payload.get("traffic_factor"), 1.0)
        auto_defaults = self.auto_payload_for_timestamp(timestamp, traffic_factor=traffic_factor)
        merged_payload = {**auto_defaults, **payload}

        temperature = _float(merged_payload.get("temperature"), 18)
        dew_point = _float(merged_payload.get("dew_point"), temperature - 3)
        humidity = _float(merged_payload.get("humidity"), 65)
        pressure = _float(merged_payload.get("pressure"), 1012)
        wind_speed = _float(merged_payload.get("wind_speed"), 2.4)
        wind_direction = _float(merged_payload.get("wind_direction"), 300)
        boundary_layer_height = max(15.0, _float(merged_payload.get("boundary_layer_height"), 260))

        wind_speed_3h = _float(merged_payload.get("wind_speed_3h"), wind_speed)
        wind_speed_6h = _float(merged_payload.get("wind_speed_6h"), wind_speed)
        wind_speed_12h = _float(merged_payload.get("wind_speed_12h"), wind_speed)
        wind_speed_24h = _float(merged_payload.get("wind_speed_24h"), wind_speed)

        boundary_layer_height_3h = max(15.0, _float(merged_payload.get("boundary_layer_height_3h"), boundary_layer_height))
        boundary_layer_height_12h = max(15.0, _float(merged_payload.get("boundary_layer_height_12h"), boundary_layer_height))

        temperature_3h = _float(merged_payload.get("temperature_3h"), temperature)
        temperature_12h = _float(merged_payload.get("temperature_12h"), temperature)
        temp_3h_ago = _float(merged_payload.get("temp_3h_ago"), temperature)
        temp_6h_ago = _float(merged_payload.get("temp_6h_ago"), temperature)

        humidity_3h = _float(merged_payload.get("humidity_3h"), humidity)
        humidity_12h = _float(merged_payload.get("humidity_12h"), humidity)

        pm25_lag1h = _float(merged_payload.get("pm25_lag1h"), 82)
        pm25_lag3h = _float(merged_payload.get("pm25_lag3h"), pm25_lag1h)
        pm25_lag6h = _float(merged_payload.get("pm25_lag6h"), pm25_lag3h)
        pm25_lag12h = _float(merged_payload.get("pm25_lag12h"), pm25_lag6h)
        pm25_lag24h = _float(merged_payload.get("pm25_lag24h"), pm25_lag12h)
        pm25_lag48h = _float(merged_payload.get("pm25_lag48h"), pm25_lag24h)

        vehicle_count = self._vehicle_count_for_year(dt.year) * max(0.75, min(1.35, traffic_factor))

        month = dt.month
        hour = dt.hour
        day_of_week = dt.weekday()
        day_of_year = dt.timetuple().tm_yday

        wdir_rad = math.radians(wind_direction)
        dew_pt_dep = temperature - dew_point

        derived = {
            "month": month,
            "hour": hour,
            "day_of_week": day_of_week,
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "month_sin": math.sin(2 * math.pi * month / 12),
            "month_cos": math.cos(2 * math.pi * month / 12),
            "doy_sin": math.sin(2 * math.pi * day_of_year / 365),
            "doy_cos": math.cos(2 * math.pi * day_of_year / 365),
            "temperature": temperature,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "humidity": humidity,
            "pressure": pressure,
            "boundary_layer_height": boundary_layer_height,
            "wdir_sin": math.sin(wdir_rad),
            "wdir_cos": math.cos(wdir_rad),
            "is_nw_wind": 1 if (wind_direction > 270 or wind_direction < 45) else 0,
            "log_wind": math.log1p(max(0.0, wind_speed)),
            "log_blh": math.log1p(max(15.0, boundary_layer_height)),
            "inv_blh": 1.0 / (max(50.0, boundary_layer_height) + 1.0),
            "cold_calm": max(0.0, 15 - temperature) * max(0.0, 2 - wind_speed),
            "inversion_proxy": ((15 - temperature) / math.log1p(max(15.0, boundary_layer_height)))
            if temperature < 15
            else 0.0,
            "hygroscopic_factor": max(0.0, min(1.0, (humidity - 60) / 40)),
            "high_humidity": 1 if humidity > 80 else 0,
            "dew_pt_dep": dew_pt_dep,
            "near_fog": 1 if dew_pt_dep < 3 else 0,
            "is_smog_season": 1 if month in (10, 11, 12, 1, 2) else 0,
            "is_winter": 1 if month in (12, 1, 2) else 0,
            "is_weekend": 1 if day_of_week >= 5 else 0,
            "is_night": 1 if (hour >= 21 or hour <= 6) else 0,
            "is_rush_hour": 1 if hour in (7, 8, 9, 17, 18, 19) else 0,
            "wind_trend": wind_speed - wind_speed_6h,
            "temp_drop_3h": temperature - temp_3h_ago,
            "temp_drop_6h": temperature - temp_6h_ago,
            "wind_speed_3h": wind_speed_3h,
            "wind_speed_6h": wind_speed_6h,
            "wind_speed_12h": wind_speed_12h,
            "wind_speed_24h": wind_speed_24h,
            "boundary_layer_height_3h": boundary_layer_height_3h,
            "boundary_layer_height_12h": boundary_layer_height_12h,
            "temperature_3h": temperature_3h,
            "temperature_12h": temperature_12h,
            "humidity_3h": humidity_3h,
            "humidity_12h": humidity_12h,
            "vehicle_count": vehicle_count,
            "log_vehicles": math.log1p(max(0.0, vehicle_count)),
            "pm25_lag1h": pm25_lag1h,
            "pm25_lag3h": pm25_lag3h,
            "pm25_lag6h": pm25_lag6h,
            "pm25_lag12h": pm25_lag12h,
            "pm25_lag24h": pm25_lag24h,
            "pm25_lag48h": pm25_lag48h,
            "pm25_roll3h": _mean([pm25_lag1h, pm25_lag3h], pm25_lag1h),
            "pm25_roll6h": _mean([pm25_lag1h, pm25_lag3h, pm25_lag6h], pm25_lag3h),
            "pm25_roll24h": _mean([pm25_lag1h, pm25_lag3h, pm25_lag6h, pm25_lag12h, pm25_lag24h], pm25_lag12h),
            "prev_hazardous": 1.0 if pm25_lag24h > 150 else 0.0,
        }

        vector = [derived[name] for name in self.features]
        auto_used = [key for key in auto_defaults if key not in payload and key not in {"timestamp", "traffic_factor"}]
        return dt, derived, np.array([vector], dtype=float), auto_used, auto_defaults

    def predict(self, payload):
        dt, feature_map, raw_vector, auto_used, auto_defaults = self.build_feature_vector(payload)
        feature_frame = pd.DataFrame(raw_vector, columns=self.features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = self.scaler.transform(feature_frame)
        scaled_frame = pd.DataFrame(scaled, columns=self.features)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_pred = float(self.model.predict(scaled_frame)[0])
            xgb_pred = float(self.xgb_model.predict(scaled_frame)[0])
            lgb_pred = float(self.lgb_model.predict(scaled_frame)[0])
        meta_input = pd.DataFrame(
            [[rf_pred, xgb_pred, lgb_pred]],
            columns=["rf", "xgb", "lgb"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack_pred = float(self.meta_model.predict(meta_input)[0])
        weighted_pred = float(0.25 * rf_pred + 0.40 * xgb_pred + 0.35 * lgb_pred)
        
        # --- Improvement C: Multiplicative bias correction (avoids clamping to 0.1) ---
        # Instead of: pred - bias  (can collapse predictions to zero)
        # We use:    pred / (1 + bias_frac)  where bias_frac = bias / max(25, pred_mean)
        hour = dt.hour
        raw_bias = self.hourly_bias_corrections.get(hour, 0.0)
        ref_level = max(25.0, weighted_pred)  # anchor to avoid division by tiny numbers
        bias_frac = raw_bias / ref_level
        # Clamp fraction to ±35% to avoid wild scaling on extreme outliers
        bias_frac = max(-0.35, min(0.35, bias_frac))
        scale = 1.0 / (1.0 + bias_frac) if bias_frac > -1.0 else 1.0
        rf_pred = max(0.1, rf_pred * scale)
        xgb_pred = max(0.1, xgb_pred * scale)
        lgb_pred = max(0.1, lgb_pred * scale)
        stack_pred = max(0.1, stack_pred * scale)
        weighted_pred = max(0.1, weighted_pred * scale)

        live_models = [
            {"name": "Random Forest", "pm25": round(rf_pred, 1)},
            {"name": "XGBoost", "pm25": round(xgb_pred, 1)},
            {"name": "LightGBM", "pm25": round(lgb_pred, 1)},
            {"name": "Weighted Avg", "pm25": round(weighted_pred, 1)},
            {"name": "Stacking Ensemble", "pm25": round(stack_pred, 1)},
        ]
        prediction = weighted_pred

        tree_preds = np.array([tree.predict(scaled)[0] for tree in self.model.estimators_], dtype=float)
        model_preds = np.array([rf_pred, xgb_pred, lgb_pred, weighted_pred, stack_pred], dtype=float)
        spread = float(max(np.std(model_preds), np.std(tree_preds) * 0.4))
        band = classify_pm25(prediction)
        low = max(0.0, prediction - 1.28 * spread)
        high = prediction + 1.28 * spread

        factor_scores = build_factor_scores(feature_map, prediction)
        drivers = explain_drivers(feature_map, prediction)
        confidence = confidence_label(spread, prediction)
        public_guidance = build_public_guidance(band, feature_map)

        return {
            "timestamp": dt.isoformat(timespec="minutes"),
            "pm25": round(prediction, 1),
            "input_mode": "Guided defaults" if auto_used else "Manual tuning",
            "auto_filled_fields": auto_used,
            "auto_defaults_preview": {
                "temperature": round(auto_defaults["temperature"], 1),
                "humidity": round(auto_defaults["humidity"], 1),
                "wind_speed": round(auto_defaults["wind_speed"], 1),
                "wind_direction": round(auto_defaults["wind_direction"], 0),
                "boundary_layer_height": round(auto_defaults["boundary_layer_height"], 0),
                "pm25_lag1h": round(auto_defaults["pm25_lag1h"], 1),
            },
            "band": band,
            "confidence": {
                "label": confidence,
                "spread": round(spread, 1),
                "low": round(low, 1),
                "high": round(high, 1),
            },
            "drivers": drivers,
            "factor_scores": factor_scores,
            "public_guidance": public_guidance,
            "narrative": build_narrative(dt, prediction, band, drivers),
            "live_model": "Weighted Avg",
            "best_saved_model": self.best_saved_model,
            "model_note": "Headline prediction uses the saved weighted ensemble because it achieved the strongest held-out R2 in the training artifacts.",
            "live_models": live_models,
            "benchmarks": self.evaluation_results,
        }


class ObservedDailyArtifacts:
    def __init__(self):
        self.scaler = joblib.load(DAILY_MODELS_DIR / "daily_scaler.pkl")
        self.features = joblib.load(DAILY_MODELS_DIR / "daily_feature_list.pkl")
        self.metrics = pd.read_csv(DAILY_MODELS_DIR / "daily_model_metrics.csv").to_dict("records")
        self.case_rows = pd.read_csv(DAILY_MODELS_DIR / "daily_comparison_cases.csv").to_dict("records")
        self.models = {
            "daily_random_forest": joblib.load(DAILY_MODELS_DIR / "daily_random_forest.pkl"),
            "daily_ridge": joblib.load(DAILY_MODELS_DIR / "daily_ridge.pkl"),
            "daily_xgboost": joblib.load(DAILY_MODELS_DIR / "daily_xgboost.pkl"),
            "daily_lightgbm": joblib.load(DAILY_MODELS_DIR / "daily_lightgbm.pkl"),
            "daily_extra_trees": joblib.load(DAILY_MODELS_DIR / "daily_extra_trees.pkl"),
        }
        for model in self.models.values():
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
        self.best_model_key = max(self.metrics, key=lambda item: item["r2"])["model_key"]
        self.model_labels = {
            "daily_random_forest": "Observed Daily Random Forest",
            "daily_ridge": "Observed Daily Ridge",
            "daily_xgboost": "Observed Daily XGBoost",
            "daily_lightgbm": "Observed Daily LightGBM",
            "daily_extra_trees": "Observed Daily Extra Trees",
            "daily_weighted_ensemble": "Observed Daily Weighted Ensemble",
        }
        self.source_df = pd.read_csv(ROOT / "lahore_air_quality_final_dataset.csv", parse_dates=["date"])
        self.monthly_defaults = (
            self.source_df.groupby("month")[["T2M", "RH2M", "WS2M", "pm25_lag_1", "pm25_lag_7"]]
            .mean()
            .round(3)
            .to_dict(orient="index")
        )

    def _auto_defaults(self, payload):
        timestamp = payload.get("timestamp") or datetime.now().strftime("%Y-%m-%dT%H:00")
        dt = datetime.fromisoformat(timestamp)
        month_defaults = self.monthly_defaults.get(
            dt.month,
            {"T2M": 25.0, "RH2M": 56.0, "WS2M": 1.5, "pm25_lag_1": 117.0, "pm25_lag_7": 116.0},
        )
        return {
            "timestamp": timestamp,
            "temperature": float(month_defaults["T2M"]),
            "humidity": float(month_defaults["RH2M"]),
            "wind_speed": float(month_defaults["WS2M"]),
            "pm25_lag_1d": float(month_defaults["pm25_lag_1"]),
            "pm25_lag_7d": float(month_defaults["pm25_lag_7"]),
        }

    def _feature_frame(self, payload):
        defaults = self._auto_defaults(payload)
        merged = {**defaults, **payload}
        dt = datetime.fromisoformat(merged["timestamp"])
        temperature = _float(merged.get("temperature"), defaults["temperature"])
        humidity = _float(merged.get("humidity"), defaults["humidity"])
        wind_speed = _float(merged.get("wind_speed"), defaults["wind_speed"])
        lag1 = _float(merged.get("pm25_lag_1d"), defaults["pm25_lag_1d"])
        lag7 = _float(merged.get("pm25_lag_7d"), defaults["pm25_lag_7d"])
        month = dt.month
        smog_season = 1 if month in (10, 11, 12, 1, 2) else 0
        day_of_week = dt.weekday()
        day_of_year = dt.timetuple().tm_yday
        week_of_year = int(dt.strftime("%V"))
        feature_map = {
            "T2M": temperature,
            "RH2M": humidity,
            "WS2M": wind_speed,
            "pm25_lag_1": lag1,
            "pm25_lag_7": lag7,
            "month": month,
            "smog_season": smog_season,
            "day_of_week": day_of_week,
            "day_of_year": day_of_year,
            "week_of_year": week_of_year,
            "month_sin": math.sin(2 * math.pi * month / 12),
            "month_cos": math.cos(2 * math.pi * month / 12),
            "doy_sin": math.sin(2 * math.pi * day_of_year / 365),
            "doy_cos": math.cos(2 * math.pi * day_of_year / 365),
            "log_pm25_lag_1": math.log1p(max(0.0, lag1)),
            "log_pm25_lag_7": math.log1p(max(0.0, lag7)),
            "humidity_wind": humidity / (wind_speed + 0.5),
            "temp_wind_ratio": temperature / (wind_speed + 0.5),
            "stagnation_proxy": max(0.0, 3.0 - wind_speed) * (humidity / 100),
            "temp_sq": temperature ** 2,
            "lag_diff": lag1 - lag7,
            "rolling_proxy": (lag1 + lag7) / 2,
        }
        frame = pd.DataFrame([[feature_map[name] for name in self.features]], columns=self.features)
        auto_used = [key for key in defaults if key not in payload]
        return dt, feature_map, frame, defaults, auto_used

    def predict(self, payload):
        requested_key = payload.get("model_key", self.best_model_key)
        dt, feature_map, frame, defaults, auto_used = self._feature_frame(payload)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = pd.DataFrame(self.scaler.transform(frame), columns=self.features)
            preds = {key: float(model.predict(scaled)[0]) for key, model in self.models.items()}
        preds["daily_weighted_ensemble"] = (
            0.45 * preds["daily_random_forest"]
            + 0.20 * preds["daily_ridge"]
            + 0.15 * preds["daily_xgboost"]
            + 0.10 * preds["daily_lightgbm"]
            + 0.10 * preds["daily_extra_trees"]
        )
        prediction = preds.get(requested_key, preds[self.best_model_key])
        model_spread = float(np.std(list(preds.values())))
        band = classify_pm25(prediction)
        drivers = []
        if feature_map["WS2M"] < 1.2:
            drivers.append("Low wind suggests weak daily ventilation.")
        if feature_map["pm25_lag_1"] > 150:
            drivers.append("Yesterday's pollution level is already high, so carryover risk remains elevated.")
        if feature_map["pm25_lag_7"] > 120:
            drivers.append("The previous week's average pattern is also elevated, which supports persistence.")
        if feature_map["smog_season"]:
            drivers.append("The selected date falls within the core smog season months.")
        if feature_map["T2M"] > 30 and feature_map["WS2M"] > 2.0:
            drivers.append("Warm air and better ventilation are pulling the daily estimate downward.")
        drivers = drivers[:4]
        public_guidance = build_public_guidance(band, {"humidity": feature_map["RH2M"], "wind_speed": feature_map["WS2M"]})
        live_models = [{"name": self.model_labels[key], "pm25": round(val, 1)} for key, val in preds.items()]
        return {
            "timestamp": dt.isoformat(timespec="minutes"),
            "pm25": round(prediction, 1),
            "band": band,
            "confidence": {
                "label": confidence_label(model_spread, prediction),
                "spread": round(model_spread, 1),
                "low": round(max(0.0, prediction - 1.28 * model_spread), 1),
                "high": round(prediction + 1.28 * model_spread, 1),
            },
            "drivers": drivers,
            "factor_scores": [
                {"label": "Yesterday's PM2.5", "value": max(5, min(100, int(feature_map["pm25_lag_1"] / 4)))},
                {"label": "Weekly PM2.5 Memory", "value": max(5, min(100, int(feature_map["pm25_lag_7"] / 4)))},
                {"label": "Humidity Pressure", "value": max(5, min(100, int((feature_map["RH2M"] - 20) * 1.2)))},
                {"label": "Ventilation", "value": max(5, min(100, int((5 - feature_map["WS2M"]) * 18)))},
                {"label": "Seasonal Pressure", "value": 85 if feature_map["smog_season"] else 35},
            ],
            "public_guidance": public_guidance,
            "narrative": (
                f"For {dt.strftime('%A %d %b %Y')}, the selected daily model estimates "
                f"{prediction:.1f} ug/m3. This daily forecast is driven mostly by recent observed "
                f"pollution memory, humidity, wind speed, and seasonal context."
            ),
            "live_model": self.model_labels.get(requested_key, requested_key),
            "best_saved_model": self.model_labels[self.best_model_key],
            "model_note": "This prediction comes from the cleaner observed daily dataset. Use it when you want more realistic season-to-season behavior.",
            "live_models": live_models,
            "benchmarks": self.metrics,
            "input_mode": "Guided defaults" if auto_used else "Manual tuning",
            "auto_filled_fields": auto_used,
            "auto_defaults_preview": {
                "temperature": round(defaults["temperature"], 1),
                "humidity": round(defaults["humidity"], 1),
                "wind_speed": round(defaults["wind_speed"], 1),
                "pm25_lag_1d": round(defaults["pm25_lag_1d"], 1),
                "pm25_lag_7d": round(defaults["pm25_lag_7d"], 1),
            },
        }


def classify_pm25(pm25):
    bands = [
        (15, "Clear", "Low immediate risk", "#8fd694"),
        (35, "Elevated", "Watch sensitive groups", "#d3e97a"),
        (55, "Caution", "Outdoor exposure should be moderated", "#f4d35e"),
        (150, "Unhealthy", "Likely discomfort for many residents", "#f28f3b"),
        (250, "Severe", "High-smog episode conditions", "#ef6351"),
    ]
    for threshold, label, guidance, color in bands:
        if pm25 <= threshold:
            return {
                "label": label,
                "guidance": guidance,
                "color": color,
                "severity": round((pm25 / 250) * 100),
            }
    return {
        "label": "Extreme",
        "guidance": "Emergency-level pollution conditions",
        "color": "#d7263d",
        "severity": 100,
    }


def confidence_label(spread, prediction):
    ratio = spread / max(25.0, prediction)
    if ratio < 0.08:
        return "High confidence"
    if ratio < 0.15:
        return "Moderate confidence"
    return "Caution: wider model spread"


def build_public_guidance(band, feature_map):
    if band["label"] == "Clear":
        return [
            "Air quality appears comparatively manageable for most people.",
            "This scenario suggests the atmosphere is ventilating the city rather than trapping emissions.",
            "It is still useful to watch trend direction if evening stagnation is expected later.",
        ]
    if band["label"] in {"Elevated", "Caution"}:
        return [
            "Sensitive groups may want to reduce longer outdoor exposure.",
            "Conditions are moving toward a buildup pattern rather than a fully clear day.",
            "If winds weaken further or humidity rises, this scenario can worsen quickly.",
        ]
    if band["label"] == "Unhealthy":
        return [
            "People may feel irritation or breathing discomfort, especially outdoors.",
            "The city is likely in a genuine smog event rather than a minor fluctuation.",
            "Short-term exposure reduction measures become sensible at this level.",
        ]
    return [
        "This looks like a severe to extreme smog episode with broad public-health relevance.",
        "Outdoor exposure should be minimized where possible, especially for children, elderly people, and those with respiratory conditions.",
        "The atmospheric setup suggests pollutants are being trapped instead of dispersed.",
    ]


def build_factor_scores(feature_map, prediction):
    stagnation = max(0, min(100, int((220 - feature_map["boundary_layer_height"]) / 2.2)))
    carryover = max(0, min(100, int(feature_map["pm25_lag1h"] / 3)))
    moisture = max(0, min(100, int((feature_map["humidity"] - 45) * 1.8)))
    transport = 70 if feature_map["is_nw_wind"] else 35
    seasonality = 88 if feature_map["is_smog_season"] else 32
    return [
        {"label": "Air Stagnation", "value": stagnation},
        {"label": "Pollution Memory", "value": carryover},
        {"label": "Moisture/Fog", "value": moisture},
        {"label": "Transport Signal", "value": transport},
        {"label": "Seasonal Pressure", "value": seasonality},
        {"label": "Overall Risk Load", "value": max(5, min(100, int(prediction / 3.2)))},
    ]


def explain_drivers(feature_map, prediction):
    drivers = []
    if feature_map["boundary_layer_height"] < 180:
        drivers.append("A shallow boundary layer is limiting vertical dispersion.")
    if feature_map["pm25_lag1h"] > 120:
        drivers.append("Recent PM2.5 carryover suggests the city is already in a polluted regime.")
    if feature_map["wind_speed"] < 2.0:
        drivers.append("Weak wind is not clearing emissions efficiently.")
    if feature_map["near_fog"] or feature_map["humidity"] > 85:
        drivers.append("Moist, near-fog conditions favor haze persistence.")
    if feature_map["is_smog_season"]:
        drivers.append("The selected date falls in Lahore's peak smog season.")
    if feature_map["is_nw_wind"]:
        drivers.append("Northwesterly flow can align with regional smoke transport.")
    if feature_map["is_rush_hour"]:
        drivers.append("Rush-hour timing can compound local emissions.")
    if prediction < 40 and feature_map["wind_speed"] > 4 and feature_map["boundary_layer_height"] > 600:
        drivers.append("Stronger wind and a deeper mixing layer are helping the city ventilate.")
    return drivers[:4]


def build_narrative(dt, prediction, band, drivers):
    time_story = "before sunrise" if dt.hour < 7 else "through the morning" if dt.hour < 12 else "into the afternoon" if dt.hour < 17 else "during the evening"
    episode_label = "clear-air window" if band["label"] == "Clear" else "building pollution pattern" if band["label"] in {"Elevated", "Caution"} else "smog episode"
    lead = (
        f"For {dt.strftime('%A %d %b %Y at %H:%M')}, the live model estimates "
        f"{prediction:.1f} ug/m3, which falls in the {band['label'].lower()} band."
    )
    detail = drivers[0] if drivers else "The signal looks balanced without a single dominant trigger."
    end = (
        f" This means demo viewers can frame the story as a {episode_label} {time_story}, "
        f"with {band['guidance'].lower()} driven mainly by atmospheric trapping and recent pollution memory."
    )
    return lead + " " + detail + end


ARTIFACTS = DemoArtifacts()
DAILY_ARTIFACTS = ObservedDailyArtifacts()


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/summary":
            return self._send_json({
                "project": {
                    "title": "Lahore Smog Intelligence Studio",
                    "subtitle": "Interactive PM2.5 demo powered by the saved climate-project models",
                    "live_model": "Weighted ensemble (real local inference)",
                    "best_saved_model": ARTIFACTS.best_saved_model,
                    "saved_models": [item["model"] for item in ARTIFACTS.evaluation_results],
                    "top_features": ARTIFACTS.top_features,
                    "benchmarks": ARTIFACTS.evaluation_results,
                    "presets": ARTIFACTS.presets,
                    "figures": ARTIFACTS.figure_cards,
                    "research_highlights": ARTIFACTS.research_highlights,
                    "glossary": ARTIFACTS.glossary,
                    "calculation_steps": ARTIFACTS.calculation_steps,
                    "guided_mode_note": "Guided Forecast mode fills weather and recent-pollution inputs using historical Lahore averages for the selected month and hour.",
                    "model_note": "Live predictions use the weighted ensemble because it had the best held-out benchmark among the saved artifacts.",
                    "daily_models": [
                        {"key": key, "label": DAILY_ARTIFACTS.model_labels[key]}
                        for key in [
                            "daily_random_forest",
                            "daily_ridge",
                            "daily_xgboost",
                            "daily_lightgbm",
                            "daily_extra_trees",
                            "daily_weighted_ensemble",
                        ]
                    ],
                    "daily_model_metrics": DAILY_ARTIFACTS.metrics,
                    "daily_cases": DAILY_ARTIFACTS.case_rows,
                    "daily_best_model": DAILY_ARTIFACTS.model_labels[DAILY_ARTIFACTS.best_model_key],
                    "daily_note": "The observed daily models are trained on the new cleaner 2022-2023 daily dataset and are better suited for season-to-season realism checks.",
                }
            })
        if route == "/" or route == "/index.html":
            return self._send_file(UI_DIR / "index.html")
        if route == "/styles.css":
            return self._send_file(UI_DIR / "styles.css")
        if route == "/app.js":
            return self._send_file(UI_DIR / "app.js")
        if route.startswith("/figures/"):
            relative = route.lstrip("/")
            return self._send_file(ROOT / relative)

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body) if body else {}
            if parsed.path == "/api/predict":
                model_key = payload.get("model_key", "")
                if str(model_key).startswith("daily_"):
                    result = DAILY_ARTIFACTS.predict(payload)
                else:
                    result = ARTIFACTS.predict(payload)
                self._send_json(result)
                return
            if parsed.path == "/api/autofill":
                timestamp = payload.get("timestamp") or datetime.now().strftime("%Y-%m-%dT%H:00")
                traffic_factor = _float(payload.get("traffic_factor"), 1.0)
                model_key = payload.get("model_key", "")
                if str(model_key).startswith("daily_"):
                    defaults = DAILY_ARTIFACTS._auto_defaults({"timestamp": timestamp})
                else:
                    defaults = ARTIFACTS.auto_payload_for_timestamp(timestamp, traffic_factor=traffic_factor)
                self._send_json({"defaults": defaults})
                return
            # --- Improvement B: Rolling multi-step future forecast with lag chaining ---
            if parsed.path == "/api/predict_range":
                from datetime import timedelta
                steps = max(1, min(int(payload.get("steps", 24)), 168))
                base_timestamp = payload.get("timestamp") or datetime.now().strftime("%Y-%m-%dT%H:00")
                base_payload = {k: v for k, v in payload.items() if k not in ("steps", "timestamp")}

                seed = ARTIFACTS.auto_payload_for_timestamp(
                    base_timestamp,
                    traffic_factor=_float(base_payload.get("traffic_factor"), 1.0),
                )
                seed.update(base_payload)

                # Rolling lag buffer: [lag1, lag3, lag6, lag12, lag24, lag48]
                lag_buffer = [
                    _float(seed.get("pm25_lag1h"), 82.0),
                    _float(seed.get("pm25_lag3h"), 82.0),
                    _float(seed.get("pm25_lag6h"), 82.0),
                    _float(seed.get("pm25_lag12h"), 82.0),
                    _float(seed.get("pm25_lag24h"), 82.0),
                    _float(seed.get("pm25_lag48h"), 82.0),
                ]

                dt_base = datetime.fromisoformat(base_timestamp)
                results_range = []
                for step_i in range(steps):
                    step_dt = dt_base + timedelta(hours=step_i)
                    step_ts = step_dt.strftime("%Y-%m-%dT%H:00")
                    step_payload = ARTIFACTS.auto_payload_for_timestamp(
                        step_ts,
                        traffic_factor=_float(base_payload.get("traffic_factor"), 1.0),
                    )
                    for k, v in base_payload.items():
                        if k != "timestamp":
                            step_payload[k] = v
                    step_payload["timestamp"] = step_ts
                    # Inject rolled-forward lag state
                    step_payload["pm25_lag1h"] = lag_buffer[0]
                    step_payload["pm25_lag3h"] = lag_buffer[1]
                    step_payload["pm25_lag6h"] = lag_buffer[2]
                    step_payload["pm25_lag12h"] = lag_buffer[3]
                    step_payload["pm25_lag24h"] = lag_buffer[4]
                    step_payload["pm25_lag48h"] = lag_buffer[5]

                    step_result = ARTIFACTS.predict(step_payload)
                    predicted_pm25 = step_result["pm25"]
                    results_range.append({
                        "step": step_i,
                        "timestamp": step_ts,
                        "pm25": predicted_pm25,
                        "band": step_result["band"],
                    })
                    # Shift lag buffer forward, insert new prediction as lag1
                    lag_buffer[5] = lag_buffer[4]
                    lag_buffer[4] = lag_buffer[3]
                    lag_buffer[3] = lag_buffer[2]
                    lag_buffer[2] = lag_buffer[1]
                    lag_buffer[1] = lag_buffer[0]
                    lag_buffer[0] = predicted_pm25

                self._send_json({"steps": results_range, "base_timestamp": base_timestamp})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )

    def _send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        data = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), DemoHandler)
    print(f"Lahore Smog demo running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
