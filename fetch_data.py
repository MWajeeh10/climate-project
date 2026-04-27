"""
climate_project/fetch_data.py
Fetches and merges data from ALL sources mentioned in the project slides.

Sources (in priority order):
  1. Open-Meteo Archive   -> Free, no key. Surface weather + BLH (2015-present)
  2. ERA5 / ECMWF CDS    -> Free account needed. Pressure-level data for REAL inversion
                            metrics: 850hPa temperature, geopotential, u/v winds
  3. NASA FIRMS           -> Free API key. Satellite fire counts (crop burning proxy)
  4. OpenAQ v3            -> Free, no key. PM2.5 from Lahore monitoring stations
  5. Kaggle fallback      -> Drop lahore_pm25_kaggle.csv in this folder

Run: python fetch_data.py
Output: lahore_merged.csv  (hourly merged dataset, 50+ features)

ERA5 download takes 10-30 min (server-side processing). All others < 5 min.
ERA5 is optional but strongly recommended for real inversion data.
"""

import os, time, warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path

warnings.filterwarnings("ignore")

LAT, LON   = 31.5204, 74.3587
START_YEAR = 2015
END_YEAR   = 2024
os.makedirs("raw_data", exist_ok=True)


# ============================================================
# SOURCE 1: Open-Meteo Historical Archive (free, no key)
# ============================================================

def fetch_openmeteo() -> pd.DataFrame:
    cache = "raw_data/openmeteo_weather.csv"
    if os.path.exists(cache):
        print("  [Open-Meteo] Loading from cache...")
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        print(f"  [Open-Meteo] {len(df):,} rows loaded")
        return df

    url = "https://archive-api.open-meteo.com/v1/archive"
    all_dfs = []
    for year in range(START_YEAR, END_YEAR + 1):
        try:
            resp = requests.get(url, params={
                "latitude":  LAT, "longitude": LON,
                "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
                "hourly": ",".join([
                    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
                    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
                    "boundary_layer_height", "precipitation", "visibility",
                ]),
                "wind_speed_unit": "ms", "timezone": "Asia/Karachi",
            }, timeout=60)
            resp.raise_for_status()
            h = resp.json()["hourly"]
            df = pd.DataFrame({
                "timestamp":             pd.to_datetime(h["time"]),
                "temperature":           h["temperature_2m"],
                "humidity":              h["relative_humidity_2m"],
                "dew_point":             h["dew_point_2m"],
                "wind_speed":            h["wind_speed_10m"],
                "wind_direction":        h["wind_direction_10m"],
                "pressure":              h["surface_pressure"],
                "boundary_layer_height": h["boundary_layer_height"],
                "precipitation":         h["precipitation"],
                "visibility":            h.get("visibility", [None]*len(h["time"])),
            })
            all_dfs.append(df)
            print(f"    {year}: {len(df):,} rows OK")
            time.sleep(0.4)
        except Exception as e:
            print(f"    {year}: FAILED ({e})")

    if not all_dfs:
        return pd.DataFrame()
    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(cache, index=False)
    print(f"  [Open-Meteo] {len(df):,} rows total -> {cache}")
    return df


# ============================================================
# SOURCE 2: ERA5 / ECMWF CDS (free, requires account setup)
# ============================================================

ERA5_SETUP = """
=== ERA5 SETUP (one-time, 10 minutes) ===
Provides: 850hPa temperature (real inversion!), geopotential,
          u/v wind at pressure levels, vertical velocity

1. Register free: https://cds.climate.copernicus.eu/user/register
2. Get API key: log in -> Profile (top-right) -> API token
3. Create file ~/.cdsapirc (Windows: C:\\Users\\YOU\\.cdsapirc):
     url: https://cds.climate.copernicus.eu/api/v2
     key: YOUR_UID:YOUR_API_KEY
     verify: 0
4. pip install cdsapi netcdf4 xarray
5. Re-run fetch_data.py

ERA5 download: 10-30 min. Data saved to raw_data/ for reuse.
==========================================
"""

def fetch_era5():
    cache = "raw_data/era5_lahore.csv"
    if os.path.exists(cache):
        print(f"  [ERA5] Loading from cache: {cache}")
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        print(f"  [ERA5] {len(df):,} rows loaded")
        return df

    try:
        import cdsapi
    except ImportError:
        print("  [ERA5] cdsapi not installed. Run: pip install cdsapi xarray netcdf4")
        print(ERA5_SETUP)
        return None

    if not (Path.home() / ".cdsapirc").exists():
        print("  [ERA5] ~/.cdsapirc not configured.")
        print(ERA5_SETUP)
        return None

    print("  [ERA5] Downloading from ECMWF CDS (10-30 min)... grab a coffee")
    c     = cdsapi.Client(quiet=True)
    years = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    area  = [32.5, 73.5, 30.5, 75.5]   # N, W, S, E (1 degree box around Lahore)

    pres_file = "raw_data/era5_pressure.nc"
    surf_file = "raw_data/era5_surface.nc"

    if not os.path.exists(pres_file):
        c.retrieve("reanalysis-era5-pressure-levels", {
            "product_type": "reanalysis", "format": "netcdf",
            "variable": [
                "temperature",           # T850: inversion detection
                "geopotential",          # Z700: circulation patterns
                "specific_humidity",     # moisture at levels
                "u_component_of_wind",   # u850: low-level jet
                "v_component_of_wind",
                "vertical_velocity",     # omega850: subsidence
            ],
            "pressure_level": ["500", "700", "850", "925"],
            "year": years,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  [f"{h:02d}:00" for h in range(24)],
            "area":  area,
        }, pres_file)

    if not os.path.exists(surf_file):
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "format": "netcdf",
            "variable": [
                "boundary_layer_height", "total_column_water_vapour",
                "2m_dewpoint_temperature", "surface_pressure",
                "total_precipitation", "soil_temperature_level_1",
            ],
            "year": years,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  [f"{h:02d}:00" for h in range(24)],
            "area":  area,
        }, surf_file)

    try:
        import xarray as xr
        ds = xr.open_dataset(pres_file).sel(
            latitude=LAT, longitude=LON, method="nearest")
        times = pd.to_datetime(
            ds["valid_time"].values if "valid_time" in ds else ds["time"].values)
        df = pd.DataFrame({
            "timestamp":         times,
            "t850":              ds["t"].sel(pressure_level=850, method="nearest").values - 273.15,
            "t500":              ds["t"].sel(pressure_level=500, method="nearest").values - 273.15,
            "z700":              ds["z"].sel(pressure_level=700, method="nearest").values / 9.81,
            "u850":              ds["u"].sel(pressure_level=850, method="nearest").values,
            "v850":              ds["v"].sel(pressure_level=850, method="nearest").values,
            "w850":              ds["w"].sel(pressure_level=850, method="nearest").values,
        })
        df["wind850"] = np.sqrt(df["u850"]**2 + df["v850"]**2)
        df["timestamp"] = df["timestamp"].dt.floor("h")
        df.to_csv(cache, index=False)
        print(f"  [ERA5] {len(df):,} rows saved to {cache}")
        return df
    except Exception as e:
        print(f"  [ERA5] Parse error: {e}")
        return None


# ============================================================
# SOURCE 3: NASA FIRMS fire counts (crop burning)
# ============================================================

FIRMS_SETUP = """
=== NASA FIRMS API KEY (free, 2 min) ===
1. Go to: https://firms.modaps.eosdis.nasa.gov/api/area/
2. Click "Get MAP_KEY" -> register with email
3. Add to .env file: NASA_FIRMS_KEY=your_key_here
   OR: export NASA_FIRMS_KEY="your_key_here"

Why: Crop burning contributes 15-25% of Lahore PM2.5 in Oct-Nov.
==========================================
"""

def fetch_firms_fires():
    cache = "raw_data/firms_fire_counts.csv"
    if os.path.exists(cache):
        print(f"  [FIRMS] Loading from cache")
        df = pd.read_csv(cache, parse_dates=["date"])
        print(f"  [FIRMS] {len(df)} daily fire records loaded")
        return df

    key = os.environ.get("NASA_FIRMS_KEY", "")
    if not key:
        # Try loading from .env file
        if os.path.exists(".env"):
            for line in open(".env"):
                if line.startswith("NASA_FIRMS_KEY="):
                    key = line.strip().split("=", 1)[1].strip('"').strip("'")
                    break
    if not key:
        print("  [FIRMS] NASA_FIRMS_KEY not set.")
        print(FIRMS_SETUP)
        return None

    # Punjab bounding box (captures all upwind crop burning)
    W, S, E, N = 71.0, 28.5, 77.5, 33.5
    rows = []
    print("  [FIRMS] Fetching satellite fire counts...")

    for year in range(START_YEAR, END_YEAR + 1):
        for month in [9, 10, 11]:   # Peak crop burning months
            date_str = f"{year}-{month:02d}-01"
            try:
                url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                       f"{key}/VIIRS_SNPP_NRT/{W},{S},{E},{N}/30/{date_str}")
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.text) > 200:
                    from io import StringIO
                    fires = pd.read_csv(StringIO(resp.text))
                    if "acq_date" in fires.columns:
                        daily = fires.groupby("acq_date").size().reset_index(name="fire_count")
                        daily.columns = ["date", "fire_count"]
                        rows.append(daily)
                        print(f"    {year}-{month:02d}: {len(daily)} fire days")
                time.sleep(0.5)
            except Exception as e:
                print(f"    {year}-{month:02d}: error ({e})")

    if not rows:
        return None
    df = pd.concat(rows).groupby("date")["fire_count"].sum().reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(cache, index=False)
    print(f"  [FIRMS] {len(df)} days saved")
    return df


# ============================================================
# SOURCE 4: OpenAQ PM2.5
# ============================================================

LAHORE_STATIONS = [
    {"id": 2178,   "name": "Gulberg"},
    {"id": 270396, "name": "US Consulate"},
    {"id": 8118,   "name": "PEPA"},
]

def fetch_openaq_pm25():
    cache = "raw_data/openaq_pm25.csv"
    if os.path.exists(cache):
        print(f"  [OpenAQ] Loading from cache")
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        print(f"  [OpenAQ] {len(df):,} hourly readings loaded")
        return df

    rows = []
    for station in LAHORE_STATIONS:
        page = 1
        print(f"  [OpenAQ] Station {station['name']}...")
        while True:
            try:
                resp = requests.get(
                    "https://api.openaq.org/v3/measurements",
                    params={"locations_id": station["id"], "parameters_id": 2,
                            "limit": 1000, "page": page,
                            "order_by": "datetime", "sort_order": "asc"},
                    headers={"Accept": "application/json"}, timeout=20,
                )
                results = resp.json().get("results", [])
                if not results:
                    break
                for r in results:
                    dt = r.get("period", {}).get("datetimeFrom", {}).get("local", "")
                    v  = r.get("value", None)
                    if dt and v is not None and 0 < v < 1500:
                        rows.append({"timestamp": pd.to_datetime(dt), "pm25": float(v)})
                page += 1
                time.sleep(0.15)
            except Exception:
                break

    if not rows:
        print("  [OpenAQ] No readings returned")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df = df.groupby("timestamp")["pm25"].mean().reset_index()
    df.to_csv(cache, index=False)
    print(f"  [OpenAQ] {len(df):,} hourly readings saved")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]

    df["month"]       = ts.dt.month
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["year"]        = ts.dt.year

    # Cyclical encodings
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)

    if "wind_direction" in df.columns:
        df["wdir_sin"]   = np.sin(np.radians(df["wind_direction"]))
        df["wdir_cos"]   = np.cos(np.radians(df["wind_direction"]))
        df["is_nw_wind"] = ((df["wind_direction"] > 270) | (df["wind_direction"] < 45)).astype(int)

    if "wind_speed" in df.columns:
        df["log_wind"]  = np.log1p(df["wind_speed"])

    if "boundary_layer_height" in df.columns:
        df["log_blh"]   = np.log1p(df["boundary_layer_height"])
        df["inv_blh"]   = 1.0 / (df["boundary_layer_height"].clip(lower=50) + 1)

    if "temperature" in df.columns and "wind_speed" in df.columns:
        cold = (15 - df["temperature"]).clip(lower=0)
        calm = (2  - df["wind_speed"] ).clip(lower=0)
        df["cold_calm"] = cold * calm

    if "temperature" in df.columns and "boundary_layer_height" in df.columns:
        df["inversion_proxy"] = np.where(
            df["temperature"] < 15,
            (15 - df["temperature"]) / np.log1p(df["boundary_layer_height"]), 0)

    # Real ERA5 inversion strength
    if "t850" in df.columns and "temperature" in df.columns:
        df["inversion_strength"] = df["t850"] - df["temperature"]
        df["strong_inversion"]   = (df["inversion_strength"] > 5).astype(int)

    if "w850" in df.columns:
        df["subsidence"] = np.where(df["w850"] < 0, np.abs(df["w850"]), 0)

    if "z700" in df.columns:
        df["z700_anom"] = df["z700"] - df["z700"].mean()

    if "humidity" in df.columns:
        df["hygroscopic_factor"] = np.clip((df["humidity"] - 60) / 40, 0, 1)
        df["high_humidity"]      = (df["humidity"] > 80).astype(int)

    if "temperature" in df.columns and "dew_point" in df.columns:
        df["dew_pt_dep"] = df["temperature"] - df["dew_point"]
        df["near_fog"]   = (df["dew_pt_dep"] < 3).astype(int)

    # Regime flags
    df["is_smog_season"] = df["month"].isin([10, 11, 12, 1, 2]).astype(int)
    df["is_winter"]      = df["month"].isin([12, 1, 2]).astype(int)
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]       = ((df["hour"] >= 21) | (df["hour"] <= 6)).astype(int)
    df["is_rush_hour"]   = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Fire features
    if "fire_count" in df.columns:
        df["fire_count"]   = df["fire_count"].fillna(0)
        df["log_fire"]     = np.log1p(df["fire_count"])
        df["fire_3d_sum"]  = df["fire_count"].rolling(72, min_periods=1).sum()
        df["log_fire_3d"]  = np.log1p(df["fire_3d_sum"])
        df["is_burn_day"]  = (df["fire_count"] > 100).astype(int)

    # Rolling weather statistics
    for col in ["wind_speed", "boundary_layer_height", "temperature", "humidity", "pressure"]:
        if col not in df.columns:
            continue
        for w, name in [(3, "3h"), (6, "6h"), (12, "12h"), (24, "24h")]:
            df[f"{col}_{name}"] = df[col].rolling(w, min_periods=1).mean()

    if "wind_speed" in df.columns:
        df["wind_trend"]    = df["wind_speed"] - df["wind_speed"].rolling(6, min_periods=1).mean()

    if "temperature" in df.columns:
        df["temp_drop_3h"]  = df["temperature"].diff(3).fillna(0)
        df["temp_drop_6h"]  = df["temperature"].diff(6).fillna(0)

    # PM2.5 lag features
    if "pm25" in df.columns:
        for lag in [1, 3, 6, 12, 24, 48]:
            df[f"pm25_lag{lag}h"] = df["pm25"].shift(lag)
        df["pm25_roll3h"]   = df["pm25"].rolling( 3, min_periods=1).mean().shift(1)
        df["pm25_roll6h"]   = df["pm25"].rolling( 6, min_periods=1).mean().shift(1)
        df["pm25_roll24h"]  = df["pm25"].rolling(24, min_periods=1).mean().shift(1)
        df["prev_hazardous"] = (df["pm25"].shift(24) > 150).astype(float)

    # Vehicle count feature (annual traffic proxy)
    if "vehicle_count" in df.columns:
        df["log_vehicles"] = np.log1p(df["vehicle_count"])

    df = df.dropna(subset=["pm25"]).reset_index(drop=True) if "pm25" in df.columns else df
    return df


# ============================================================
# PHYSICS PM2.5 (fallback only)
# ============================================================

def physics_pm25(df: pd.DataFrame) -> np.ndarray:
    n    = len(df)
    mon  = df["timestamp"].dt.month.values
    hour = df["timestamp"].dt.hour.values
    monthly = {1:185, 2:120, 3:75, 4:45, 5:35, 6:30, 7:28, 8:32, 9:42, 10:95, 11:220, 12:250}
    base = np.array([monthly[m] for m in mon], dtype=float)

    blh  = np.clip(df.get("boundary_layer_height", pd.Series(np.full(n, 400.0))).values, 50, 2000)
    ws   = np.clip(df.get("wind_speed",             pd.Series(np.full(n, 2.0))).values,   0, 15)
    temp = df.get("temperature",  pd.Series(np.full(n, 15.0))).values
    hum  = df.get("humidity",     pd.Series(np.full(n, 70.0))).values
    pres = df.get("pressure",     pd.Series(np.full(n, 1013.0))).values
    wdir = df.get("wind_direction", pd.Series(np.full(n, 315.0))).values
    fire = df.get("fire_count",   pd.Series(np.zeros(n))).fillna(0).values

    # ERA5 inversion enhancement
    t850 = df.get("t850", pd.Series(np.full(n, float("nan")))).values
    inv_boost = np.where(
        ~np.isnan(t850) & (t850 > temp),
        3.0 * np.clip(t850 - temp, 0, 20),   # real inversion data
        0)

    pm25 = (base
        + 200 * np.exp(-blh / 260)
        - 52  * np.log1p(ws)
        + np.where(temp < 15, -5.5 * (temp - 15), 0)
        + 1.0 * np.clip(hum - 55, 0, 44)
        + 0.45 * (pres - 1013)
        + 20  * np.where((wdir > 270) | (wdir < 45), np.abs(np.cos(np.radians(wdir - 315))), 0)
        + 0.8 * np.sqrt(fire + 1)
        + inv_boost
        + (35 * np.exp(-0.5 * ((hour - 5.5) / 1.5)**2)
         + 22 * np.exp(-0.5 * ((hour - 21)  / 2.0)**2))
        + np.random.normal(0, 5, n))
    return np.clip(pm25, 2, 900).round(1)


# ============================================================
# VEHICLE DATA MERGE
# ============================================================

def merge_vehicle_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merge annual vehicle count from raw_data/vehicle_data.csv by year."""
    veh_path = "raw_data/vehicle_data.csv"
    if not os.path.exists(veh_path):
        print("  [Vehicles] raw_data/vehicle_data.csv not found — skipping")
        return df

    veh = pd.read_csv(veh_path)
    veh.columns = [c.strip().lower() for c in veh.columns]

    # Clean comma-formatted numbers if present
    if "vehicle_count" not in veh.columns and "vehicles" in veh.columns:
        veh["vehicle_count"] = pd.to_numeric(
            veh["vehicles"].astype(str).str.replace(",", ""), errors="coerce")

    if "vehicle_count" not in veh.columns or "year" not in veh.columns:
        print("  [Vehicles] Missing year/vehicle_count columns — skipping")
        return df

    known = veh.set_index("year")["vehicle_count"].dropna()
    if known.empty:
        return df

    # Linear trend for extrapolation
    slope = (known.iloc[-1] - known.iloc[0]) / max(
        1, known.index[-1] - known.index[0])

    df["year"] = df["timestamp"].dt.year
    df = pd.merge(df, veh[["year", "vehicle_count"]], on="year", how="left")

    # Extrapolate for years outside 2019-2023
    for yr in df[df["vehicle_count"].isna()]["year"].unique():
        nearest_yr = known.index[np.argmin(np.abs(known.index - yr))]
        extrap = known[nearest_yr] + slope * (yr - nearest_yr)
        df.loc[df["year"] == yr, "vehicle_count"] = extrap

    print(f"  [Vehicles] vehicle_count merged for years "
          f"{df.year.min()}–{df.year.max()} "
          f"(range: {df.vehicle_count.min():,.0f}–{df.vehicle_count.max():,.0f})")
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  Lahore PM2.5 - Comprehensive Multi-Source Data Fetcher")
    print("=" * 65)

    print("\n[1/5] Open-Meteo: surface weather...")
    weather = fetch_openmeteo()
    if weather is None or weather.empty:
        print("FATAL: weather data unavailable"); return
    weather["timestamp"] = weather["timestamp"].dt.floor("h")

    print("\n[2/5] ERA5 ECMWF: pressure-level inversion data...")
    era5 = fetch_era5()

    print("\n[3/5] NASA FIRMS: satellite fire counts...")
    fires = fetch_firms_fires()

    print("\n[4/5] OpenAQ: PM2.5 measurements...")
    pm25_df = fetch_openaq_pm25()

    # Kaggle supplement
    if os.path.exists("lahore_pm25_kaggle.csv"):
        print("  Found lahore_pm25_kaggle.csv - merging...")
        kdf = pd.read_csv("lahore_pm25_kaggle.csv")
        kdf.columns = [c.lower().strip().replace(" ", "_") for c in kdf.columns]
        ts_col = next((c for c in kdf.columns if "date" in c or "time" in c), None)
        pm_col = next((c for c in kdf.columns if "pm" in c and "2" in c), None)
        if ts_col and pm_col:
            kdf = kdf[[ts_col, pm_col]].rename(columns={ts_col: "timestamp", pm_col: "pm25"})
            kdf["timestamp"] = pd.to_datetime(kdf["timestamp"]).dt.floor("h")
            kdf["pm25"] = pd.to_numeric(kdf["pm25"], errors="coerce").clip(0, 1500)
            kdf = kdf.dropna()
            pm25_df = pd.concat([pm25_df, kdf]).groupby("timestamp")["pm25"].mean().reset_index()
            print(f"  After Kaggle merge: {len(pm25_df):,} PM2.5 readings")

    print("\n[5/5] Merging all sources and engineering features...")
    merged = weather.copy()

    if era5 is not None:
        era5["timestamp"] = pd.to_datetime(era5["timestamp"]).dt.floor("h")
        merged = pd.merge(merged, era5, on="timestamp", how="left")
        print(f"  ERA5: {era5.shape[1]-1} inversion columns added")

    if fires is not None:
        fires["date"] = pd.to_datetime(fires["date"]).dt.date
        merged["date"] = merged["timestamp"].dt.date
        merged = pd.merge(merged, fires, on="date", how="left")
        merged["fire_count"] = merged["fire_count"].fillna(0)
        merged.drop("date", axis=1, inplace=True)
        print(f"  FIRMS: fire_count column added")

    if not pm25_df.empty:
        merged = pd.merge(merged, pm25_df, on="timestamp", how="inner")
        print(f"  Rows with REAL PM2.5: {len(merged):,}")
    else:
        print("  Generating physics-based PM2.5...")
        merged["pm25"] = physics_pm25(merged)

    # Merge vehicle count data
    merged = merge_vehicle_data(merged)

    merged = engineer_features(merged)

    print(f"\n  Final dataset: {len(merged):,} rows x {len(merged.columns)} columns")
    print(f"  PM2.5 mean={merged.pm25.mean():.1f} | "
          f"min={merged.pm25.min():.1f} | max={merged.pm25.max():.1f}")
    print(f"  Hazardous hours (>150): {(merged.pm25>150).sum():,} "
          f"({(merged.pm25>150).mean()*100:.1f}%)")
    print(f"  Date range: {merged.timestamp.min()} to {merged.timestamp.max()}")

    merged.to_csv("lahore_merged.csv", index=False)
    print(f"\n  Saved: lahore_merged.csv")

    print("\nDATA SOURCES SUMMARY:")
    print(f"  Open-Meteo weather : OK ({len(weather):,} rows)")
    print(f"  ERA5 inversions    : {'OK' if era5 is not None else 'SKIPPED - see ERA5_SETUP above'}")
    print(f"  NASA FIRMS fires   : {'OK' if fires is not None else 'SKIPPED - set NASA_FIRMS_KEY'}")
    print(f"  OpenAQ PM2.5       : {'OK - REAL DATA' if not pm25_df.empty else 'SKIPPED - using physics-based'}")

if __name__ == "__main__":
    main()
