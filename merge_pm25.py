"""
climate_project/merge_pm25.py

Merges ALL PM2.5 data sources into one clean hourly series covering
Jun 2019 → Apr 15 2026.

Source files and their coverage:
  lahore_aqi_2019_to_2023.csv   Jun 2019 – Nov 2023  (daily AQI, imperial units)
  dataset_part_2.csv            Nov 2025 – Feb 2026  (hourly PM2.5, multi-city)
  raw_data/epd_aqi_scraped.csv  Dec 2023 – Apr 2026  (daily, scraped PDFs)
  raw_data/openaq_pm25.csv      varies               (hourly, where available)

Output:
  raw_data/pm25_merged_daily.csv   — one row per day, clean PM2.5
  raw_data/pm25_merged_final.csv   — expanded to hourly using diurnal profile

Run: python merge_pm25.py
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

os.makedirs("raw_data", exist_ok=True)

END_DATE  = pd.Timestamp("2026-04-15")
LAHORE_LAT, LAHORE_LON = 31.5204, 74.3587

# AQI → PM2.5 EPA breakpoints
AQI_BP = [
    (0,   50,  0.0,   12.0),
    (51,  100, 12.1,  35.4),
    (101, 150, 35.5,  55.4),
    (151, 200, 55.5,  150.4),
    (201, 300, 150.5, 250.4),
    (301, 400, 250.5, 350.4),
    (401, 500, 350.5, 500.4),
]

def aqi_to_pm25(aqi):
    if pd.isna(aqi) or aqi < 0:
        return None
    for i_lo, i_hi, c_lo, c_hi in AQI_BP:
        if i_lo <= aqi <= i_hi:
            return round((c_hi - c_lo) / (i_hi - i_lo) * (aqi - i_lo) + c_lo, 1)
    return None


# ════════════════════════════════════════════════════════════════════════
# SOURCE 1: lahore_aqi_2019_to_2023.csv
# Daily AQI + weather in imperial units. Jun 2019 – Nov 2023.
# ════════════════════════════════════════════════════════════════════════

def load_aqi_2019_2023(path: str = "lahore_aqi_2019_to_2023.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  [AQI 2019-2023] File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("\ufeff").lower().replace(" ", "_")
                  for c in df.columns]

    # Parse date — format is DD-MM-YY
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%y", errors="coerce")
    df = df.dropna(subset=["date"])

    # AQI → PM2.5
    df["aqi_val"] = pd.to_numeric(df["aqi_pm2.5"], errors="coerce")
    df["pm25"]    = df["aqi_val"].apply(aqi_to_pm25)

    # Convert weather columns from imperial to metric for later use in fetch_data
    # Temperature: F → C
    for col in ["max_temp_f", "avg_temp_f", "min_temp_f"]:
        if col in df.columns:
            df[col.replace("_f", "_c")] = (
                pd.to_numeric(df[col], errors="coerce") - 32) * 5 / 9

    # Dew point: F → C
    for col in ["max_dew_point_f", "avg_dew_point_f", "min_dew_point_f"]:
        if col in df.columns:
            df[col.replace("_f", "_c")] = (
                pd.to_numeric(df[col], errors="coerce") - 32) * 5 / 9

    # Wind: mph → m/s
    for col in ["max_wind_speed_mph", "avg_wind_speed_mph", "min_wind_speed_mph"]:
        if col in df.columns:
            df[col.replace("_mph", "_ms")] = (
                pd.to_numeric(df[col], errors="coerce") * 0.44704)

    # Pressure: inHg → hPa
    for col in ["max_pressure_in", "avg_pressure_in", "min_pressure_in"]:
        if col in df.columns:
            df[col.replace("_in", "_hpa")] = (
                pd.to_numeric(df[col], errors="coerce") * 33.8639)

    # Humidity: already in %
    for col in ["avg_humidity_percent"]:
        if col in df.columns:
            df["humidity"] = pd.to_numeric(df[col], errors="coerce")

    result = df[df["pm25"].notna()].copy()
    result["source"] = "aqi_2019_2023"

    print(f"  [AQI 2019-2023] {len(result):,} daily rows | "
          f"{result.date.min().date()} → {result.date.max().date()} | "
          f"PM2.5 mean={result.pm25.mean():.1f}")
    return result[["date", "pm25", "source",
                   "avg_temp_c", "humidity",
                   "avg_wind_speed_ms", "avg_pressure_hpa",
                   "avg_dew_point_c"]].rename(columns={
        "avg_temp_c":      "temperature",
        "avg_wind_speed_ms": "wind_speed",
        "avg_pressure_hpa":  "pressure",
        "avg_dew_point_c":   "dew_point",
    })


# ════════════════════════════════════════════════════════════════════════
# SOURCE 2: dataset_part_2.csv
# Hourly, multi-city. Filter Lahore. Nov 2025 – Feb 2026.
# Already has PM2.5 and all weather columns in metric.
# ════════════════════════════════════════════════════════════════════════

def load_dataset_part2(path: str = "dataset_part_2.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  [dataset_part_2] File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Filter to Lahore
    city_col = next((c for c in df.columns if "city" in c.lower()), None)
    if city_col:
        df = df[df[city_col].str.lower() == "lahore"].copy()

    # Parse timestamp
    ts_col  = next((c for c in df.columns
                    if any(k in c.lower() for k in ["timestamp", "datetime", "date"])), None)
    pm_col  = next((c for c in df.columns
                    if re.search(r"pm2[_\s]?5|pm25", c, re.I)), None)

    if ts_col is None or pm_col is None:
        print(f"  [dataset_part_2] Could not find timestamp or PM2.5 column")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["pm25"]      = pd.to_numeric(df[pm_col], errors="coerce")
    df["date"]      = df["timestamp"].dt.floor("D")
    df["source"]    = "dataset_part_2"

    # Pull in weather columns if present (already metric)
    weather_map = {
        "temperature": ["temperature", "temp"],
        "humidity":    ["humidity", "relative_humidity"],
        "wind_speed":  ["wind_speed"],
        "pressure":    ["pressure"],
        "wind_direction": ["wind_direction"],
    }
    for target, candidates in weather_map.items():
        for cand in candidates:
            match = next((c for c in df.columns
                          if c.lower().replace(" ", "_") == cand), None)
            if match:
                df[target] = pd.to_numeric(df[match], errors="coerce")
                break

    df = df.dropna(subset=["timestamp", "pm25"])
    df = df[df["pm25"].between(0.5, 1500)]

    print(f"  [dataset_part_2] {len(df):,} hourly Lahore rows | "
          f"{df.timestamp.min().date()} → {df.timestamp.max().date()} | "
          f"PM2.5 mean={df.pm25.mean():.1f}")
    return df


# ════════════════════════════════════════════════════════════════════════
# SOURCE 3: EPD Punjab scraped (daily)
# ════════════════════════════════════════════════════════════════════════

def load_epd_scraped(path: str = "raw_data/epd_aqi_scraped.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  [EPD scraped] Not found — run scrape_epd.py first")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df.get("date", df.get("timestamp")),
                                 errors="coerce")
    df["pm25"] = pd.to_numeric(df.get("pm25", df.get("pm2_5")), errors="coerce")
    df = df.dropna(subset=["date", "pm25"])
    df = df[df["pm25"].between(0.5, 1500)]
    df["source"] = "epd_scraped"

    print(f"  [EPD scraped]   {len(df):,} daily rows | "
          f"{df.date.min().date()} → {df.date.max().date()} | "
          f"PM2.5 mean={df.pm25.mean():.1f}")
    return df[["date", "pm25", "source"]]


# ════════════════════════════════════════════════════════════════════════
# SOURCE 4: OpenAQ (hourly)
# ════════════════════════════════════════════════════════════════════════

def load_openaq(path: str = "raw_data/openaq_pm25.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"]   = df["timestamp"].dt.floor("D")
    df["source"] = "openaq"
    df = df[df["pm25"].between(0.5, 1500)]

    print(f"  [OpenAQ]        {len(df):,} hourly rows | "
          f"{df.date.min().date()} → {df.date.max().date()} | "
          f"PM2.5 mean={df.pm25.mean():.1f}")
    return df


# ════════════════════════════════════════════════════════════════════════
# SOURCE 5: lahore_other_2019_to_2023.csv
# Yearly population + vehicle counts — used as annual features, not PM2.5
# ════════════════════════════════════════════════════════════════════════

def load_vehicle_data(path: str = "lahore_other_2019_to_2023.csv") -> pd.DataFrame:
    """Returns a year → vehicle_count lookup for use as a feature."""
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["vehicles_clean"] = (df["vehicles"].astype(str)
                            .str.replace(",", "", regex=False)
                            .str.strip())
    df["vehicle_count"]  = pd.to_numeric(df["vehicles_clean"], errors="coerce")
    df["population_clean"] = (df["population"].astype(str)
                              .str.replace(",", "", regex=False))
    df["population_count"] = pd.to_numeric(df["population_clean"], errors="coerce")

    print(f"  [Vehicle data]  {len(df)} years: "
          f"{df.year.min()}–{df.year.max()} | "
          f"vehicles {df.vehicle_count.min():,.0f}–{df.vehicle_count.max():,.0f}")
    return df[["year", "vehicle_count", "population_count"]]


# ════════════════════════════════════════════════════════════════════════
# MERGE + DEDUPLICATE
# ════════════════════════════════════════════════════════════════════════

# Priority weights: higher = trusted more when sources conflict
PRIORITY = {
    "openaq":          1.0,
    "dataset_part_2":  1.0,
    "aqi_2019_2023":   0.95,   # real sensor but AQI→PM2.5 conversion adds small error
    "epd_scraped":     0.80,   # PDF extraction, sometimes imprecise
}

def merge_to_daily(frames: list) -> pd.DataFrame:
    """
    Merge all sources into one daily series.
    For days covered by multiple sources, take the priority-weighted average.
    """
    rows = []
    for df in frames:
        if df.empty:
            continue
        if "date" not in df.columns and "timestamp" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["timestamp"]).dt.floor("D")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
        df["weight"] = df["source"].map(
            lambda s: next((v for k, v in PRIORITY.items()
                            if s.startswith(k)), 0.85)
        )
        rows.append(df[["date", "pm25", "weight", "source"]])

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.dropna(subset=["pm25"])
    combined = combined[combined["pm25"].between(0.5, 1500)]
    combined = combined[combined["date"] <= END_DATE]

    def wavg(g):
        w = g["weight"].values
        v = g["pm25"].values
        return pd.Series({
            "pm25":      round(float(np.average(v, weights=w)), 1),
            "n_sources": len(g),
            "sources":   "|".join(g["source"].unique()),
        })

    daily = (combined.groupby("date")
             .apply(wavg)
             .reset_index()
             .sort_values("date")
             .reset_index(drop=True))

    return daily


def fill_gaps(daily: pd.DataFrame) -> pd.DataFrame:
    full = pd.DataFrame({
        "date": pd.date_range(daily.date.min(), END_DATE, freq="D")
    })
    daily = pd.merge(full, daily, on="date", how="left")

    missing = daily[daily["pm25"].isna()]
    if not missing.empty:
        # Print gap summary
        gaps  = (missing["date"].diff().dt.days.fillna(1) != 1).cumsum()
        print(f"\n  Gaps ({len(missing)} days):")
        for _, g in missing.groupby(gaps):
            print(f"    {g.date.min().date()} → {g.date.max().date()} ({len(g)} d)")

    # Interpolate gaps up to 7 consecutive days
    daily["pm25"] = daily["pm25"].interpolate(method="linear", limit=7)
    filled = daily["pm25"].notna().sum()
    print(f"\n  After interpolation: {filled}/{len(daily)} days have data")
    return daily


# ════════════════════════════════════════════════════════════════════════
# DIURNAL EXPANSION → HOURLY
# ════════════════════════════════════════════════════════════════════════

# Lahore documented diurnal multipliers (from literature)
# Peak at 05:00–07:00 (pre-dawn stagnation), trough at 13:00–14:00
DIURNAL = {
     0: 1.15,  1: 1.20,  2: 1.25,  3: 1.28,  4: 1.30,
     5: 1.35,  6: 1.35,  7: 1.25,  8: 1.20,  9: 1.10,
    10: 1.00, 11: 0.92, 12: 0.85, 13: 0.80, 14: 0.78,
    15: 0.82, 16: 0.90, 17: 1.00, 18: 1.10, 19: 1.18,
    20: 1.20, 21: 1.18, 22: 1.15, 23: 1.12,
}
# Normalise so mean = 1.0
_mean = sum(DIURNAL.values()) / 24
DIURNAL = {h: v / _mean for h, v in DIURNAL.items()}


def expand_to_hourly(daily: pd.DataFrame,
                     part2_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a full hourly PM2.5 series.
    - Where dataset_part_2 has real hourly data → use that directly.
    - Elsewhere → expand daily using diurnal profile.
    """
    # Index real hourly data by timestamp
    if not part2_hourly.empty:
        p2 = part2_hourly[["timestamp", "pm25"]].copy()
        p2["timestamp"] = pd.to_datetime(p2["timestamp"]).dt.floor("h")
        real_hourly = p2.set_index("timestamp")["pm25"].to_dict()
    else:
        real_hourly = {}

    rows = []
    for _, row in daily.iterrows():
        d = pd.to_datetime(row["date"])
        for hour in range(24):
            ts = d.replace(hour=hour)
            if ts in real_hourly:
                pm25 = real_hourly[ts]
            elif pd.notna(row["pm25"]):
                pm25 = max(1.0, round(row["pm25"] * DIURNAL[hour], 1))
            else:
                continue
            rows.append({
                "timestamp": ts,
                "pm25":      round(float(pm25), 1),
                "source":    row.get("sources", "merged"),
            })

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PM2.5 Multi-Source Merger")
    print("=" * 60)

    print("\n[1/6] Loading lahore_aqi_2019_to_2023.csv ...")
    aqi_df = load_aqi_2019_2023()

    print("\n[2/6] Loading dataset_part_2.csv ...")
    p2_df  = load_dataset_part2()

    print("\n[3/6] Loading EPD scraped data ...")
    epd_df = load_epd_scraped()

    print("\n[4/6] Loading OpenAQ ...")
    oaq_df = load_openaq()

    print("\n[5/6] Loading vehicle/population data ...")
    veh_df = load_vehicle_data()
    if not veh_df.empty:
        veh_df.to_csv("raw_data/vehicle_data.csv", index=False)
        print("  Saved to raw_data/vehicle_data.csv (for use as annual feature)")

    print("\n[6/6] Merging all sources ...")

    # For daily merge, gather all frames (convert hourly → daily first)
    frames = []
    for df, label in [(aqi_df, "aqi"), (epd_df, "epd"), (oaq_df, "oaq")]:
        if not df.empty:
            frames.append(df)

    # dataset_part_2 daily aggregate for the merge
    if not p2_df.empty:
        p2_daily = (p2_df.groupby("date")["pm25"]
                    .mean().reset_index()
                    .rename(columns={"pm25": "pm25"}))
        p2_daily["source"] = "dataset_part_2"
        frames.append(p2_daily)

    daily  = merge_to_daily(frames)
    daily  = fill_gaps(daily)

    print("\n  Expanding to hourly ...")
    hourly = expand_to_hourly(daily, p2_df)

    # Save
    daily_path  = "raw_data/pm25_merged_daily.csv"
    hourly_path = "raw_data/pm25_merged_final.csv"
    daily.to_csv(daily_path,   index=False)
    hourly.to_csv(hourly_path, index=False)

    # Summary
    valid = daily.dropna(subset=["pm25"])
    print(f"\n{'='*60}")
    print(f"  COVERAGE SUMMARY")
    print(f"{'='*60}")
    print(f"  Date range:      {valid.date.min().date()} → {valid.date.max().date()}")
    print(f"  Daily rows:      {len(valid):,}")
    print(f"  Hourly rows:     {len(hourly):,}")
    print(f"  PM2.5 mean:      {valid.pm25.mean():.1f} µg/m³")
    print(f"  PM2.5 max:       {valid.pm25.max():.1f} µg/m³")
    haz = (valid.pm25 > 150).mean() * 100
    print(f"  Hazardous days:  {haz:.1f}%  (>150 µg/m³)")
    print(f"\n  Saved:")
    print(f"    {daily_path}")
    print(f"    {hourly_path}")
    print(f"\n  Run next: python fetch_data.py")


if __name__ == "__main__":
    main()
