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


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
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
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.features = joblib.load(MODELS_DIR / "feature_list_full.pkl")
        self.vehicle_lookup = self._load_vehicle_lookup()
        self.evaluation_results = self._load_evaluation_results()
        self.top_features = self._load_top_features()
        self.figure_cards = self._load_figure_cards()
        self.presets = self._build_presets()

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
                "caption": "How closely the saved models follow actual smog spikes on the test set.",
                "path": "/figures/predictions.png",
            },
            {
                "title": "Model Fit Scatter",
                "caption": "Predicted vs actual PM2.5 points for the trained models.",
                "path": "/figures/scatter.png",
            },
            {
                "title": "Feature Story",
                "caption": "Top features learned by the training pipeline.",
                "path": "/figures/feature_importance.png",
            },
            {
                "title": "Explainability",
                "caption": "SHAP summary view of what pushes smog risk upward or downward.",
                "path": "/figures/shap_beeswarm.png",
            },
            {
                "title": "Calibration",
                "caption": "How well hazardous-episode risk aligns with observed outcomes.",
                "path": "/figures/calibration_curve.png",
            },
        ]
        return [card for card in cards if (ROOT / card["path"].lstrip("/")).exists()]

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

        temperature = _float(payload.get("temperature"), 18)
        dew_point = _float(payload.get("dew_point"), temperature - 3)
        humidity = _float(payload.get("humidity"), 65)
        pressure = _float(payload.get("pressure"), 1012)
        wind_speed = _float(payload.get("wind_speed"), 2.4)
        wind_direction = _float(payload.get("wind_direction"), 300)
        boundary_layer_height = max(15.0, _float(payload.get("boundary_layer_height"), 260))

        wind_speed_3h = _float(payload.get("wind_speed_3h"), wind_speed)
        wind_speed_6h = _float(payload.get("wind_speed_6h"), wind_speed)
        wind_speed_12h = _float(payload.get("wind_speed_12h"), wind_speed)
        wind_speed_24h = _float(payload.get("wind_speed_24h"), wind_speed)

        boundary_layer_height_3h = max(15.0, _float(payload.get("boundary_layer_height_3h"), boundary_layer_height))
        boundary_layer_height_12h = max(15.0, _float(payload.get("boundary_layer_height_12h"), boundary_layer_height))

        temperature_3h = _float(payload.get("temperature_3h"), temperature)
        temperature_12h = _float(payload.get("temperature_12h"), temperature)
        temp_3h_ago = _float(payload.get("temp_3h_ago"), temperature)
        temp_6h_ago = _float(payload.get("temp_6h_ago"), temperature)

        humidity_3h = _float(payload.get("humidity_3h"), humidity)
        humidity_12h = _float(payload.get("humidity_12h"), humidity)

        pm25_lag1h = _float(payload.get("pm25_lag1h"), 82)
        pm25_lag3h = _float(payload.get("pm25_lag3h"), pm25_lag1h)
        pm25_lag6h = _float(payload.get("pm25_lag6h"), pm25_lag3h)
        pm25_lag12h = _float(payload.get("pm25_lag12h"), pm25_lag6h)
        pm25_lag24h = _float(payload.get("pm25_lag24h"), pm25_lag12h)
        pm25_lag48h = _float(payload.get("pm25_lag48h"), pm25_lag24h)

        traffic_factor = _float(payload.get("traffic_factor"), 1.0)
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
        return dt, derived, np.array([vector], dtype=float)

    def predict(self, payload):
        dt, feature_map, raw_vector = self.build_feature_vector(payload)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = self.scaler.transform(raw_vector)
        prediction = float(self.model.predict(scaled)[0])

        tree_preds = np.array([tree.predict(scaled)[0] for tree in self.model.estimators_], dtype=float)
        spread = float(np.std(tree_preds))
        band = classify_pm25(prediction)
        low = max(0.0, prediction - 1.28 * spread)
        high = prediction + 1.28 * spread

        factor_scores = build_factor_scores(feature_map, prediction)
        drivers = explain_drivers(feature_map, prediction)
        confidence = confidence_label(spread, prediction)

        return {
            "timestamp": dt.isoformat(timespec="minutes"),
            "pm25": round(prediction, 1),
            "band": band,
            "confidence": {
                "label": confidence,
                "spread": round(spread, 1),
                "low": round(low, 1),
                "high": round(high, 1),
            },
            "drivers": drivers,
            "factor_scores": factor_scores,
            "narrative": build_narrative(dt, prediction, band, drivers),
            "live_model": "Random Forest",
            "benchmarks": self.evaluation_results,
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


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/summary":
            return self._send_json({
                "project": {
                    "title": "Lahore Smog Intelligence Studio",
                    "subtitle": "Interactive PM2.5 demo powered by the saved climate-project models",
                    "live_model": "Random Forest (real local inference)",
                    "saved_models": [item["model"] for item in ARTIFACTS.evaluation_results],
                    "top_features": ARTIFACTS.top_features,
                    "benchmarks": ARTIFACTS.evaluation_results,
                    "presets": ARTIFACTS.presets,
                    "figures": ARTIFACTS.figure_cards,
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
        if parsed.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body) if body else {}
            result = ARTIFACTS.predict(payload)
            self._send_json(result)
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
