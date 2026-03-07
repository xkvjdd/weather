import os
import re
import math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATA_DIR = REPO_ROOT / "data" / "dashboard"
MODEL_DIR = REPO_ROOT / "model"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OBS_MODEL_PATH = DATA_DIR / "observations_model.parquet"
FORECAST_MODEL_PATH = DATA_DIR / "forecast_model.parquet"
PEAK_OFFSET_PATH = MODEL_DIR / "peak_offset_by_city_month.csv"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT = 30

PAST_DAYS = 3
FORECAST_DAYS = 2


# ---------------------------------------------------------------------
# AIRPORT METADATA
# city should match peak_offset_by_city_month.csv
# ---------------------------------------------------------------------
AIRPORTS = {
    "ATL": {"city": "ATL", "icao": "KATL", "lat": 33.6407, "lon": -84.4277, "tz": "America/New_York", "is_us_airport": 1, "is_coastal_airport": 0},
    "NYC": {"city": "NYC", "icao": "KLGA", "lat": 40.7769, "lon": -73.8740, "tz": "America/New_York", "is_us_airport": 1, "is_coastal_airport": 1},
    "CHI": {"city": "CHI", "icao": "KORD", "lat": 41.9742, "lon": -87.9073, "tz": "America/Chicago", "is_us_airport": 1, "is_coastal_airport": 0},
    "DAL": {"city": "DAL", "icao": "KDAL", "lat": 32.8471, "lon": -96.8517, "tz": "America/Chicago", "is_us_airport": 1, "is_coastal_airport": 0},
    "SEA": {"city": "SEA", "icao": "KSEA", "lat": 47.4502, "lon": -122.3088, "tz": "America/Los_Angeles", "is_us_airport": 1, "is_coastal_airport": 1},
    "MIA": {"city": "MIA", "icao": "KMIA", "lat": 25.7959, "lon": -80.2870, "tz": "America/New_York", "is_us_airport": 1, "is_coastal_airport": 1},
    "TOR": {"city": "TOR", "icao": "CYYZ", "lat": 43.6777, "lon": -79.6248, "tz": "America/Toronto", "is_us_airport": 0, "is_coastal_airport": 0},
    "PAR": {"city": "PAR", "icao": "LFPG", "lat": 49.0097, "lon": 2.5479, "tz": "Europe/Paris", "is_us_airport": 0, "is_coastal_airport": 0},
    "SEL": {"city": "SEL", "icao": "RKSI", "lat": 37.4602, "lon": 126.4407, "tz": "Asia/Seoul", "is_us_airport": 0, "is_coastal_airport": 1},
    "ANK": {"city": "ANK", "icao": "LTAC", "lat": 40.1281, "lon": 32.9951, "tz": "Europe/Istanbul", "is_us_airport": 0, "is_coastal_airport": 0},
    "BUE": {"city": "BUE", "icao": "SAEZ", "lat": -34.8222, "lon": -58.5358, "tz": "America/Argentina/Buenos_Aires", "is_us_airport": 0, "is_coastal_airport": 1},
    "LON": {"city": "LON", "icao": "EGLL", "lat": 51.4700, "lon": -0.4543, "tz": "Europe/London", "is_us_airport": 0, "is_coastal_airport": 1},
    "WLG": {"city": "WLG", "icao": "NZWN", "lat": -41.3272, "lon": 174.8050, "tz": "Pacific/Auckland", "is_us_airport": 0, "is_coastal_airport": 1},
}


# ---------------------------------------------------------------------
# JOBLIB LOAD COMPAT
# ---------------------------------------------------------------------
def cast_to_float32(X):
    try:
        return X.astype("float32")
    except Exception:
        return X


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_peak_offsets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing peak offset file: {path}")
    df = pd.read_csv(path)
    required = {"city", "month", "mean_peak_offset_hours"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"peak_offset_by_city_month.csv missing columns: {sorted(missing)}")
    out = df[["city", "month", "mean_peak_offset_hours"]].copy()
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    out["mean_peak_offset_hours"] = pd.to_numeric(out["mean_peak_offset_hours"], errors="coerce")
    return out.dropna(subset=["city", "month", "mean_peak_offset_hours"]).copy()


def discover_models(model_dir: Path) -> dict:
    models = {}
    pat = re.compile(r"03_model_h(\d+)_(mean|q50|q90|q95)\.joblib$", re.I)

    for path in model_dir.glob("03_model_h*_*.joblib"):
        m = pat.match(path.name)
        if not m:
            continue
        horizon = int(m.group(1))
        kind = m.group(2).lower()
        models.setdefault(horizon, {})
        models[horizon][kind] = joblib.load(path)

    if not models:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    return models


def get_required_columns_from_models(models: dict) -> list[str]:
    first_h = sorted(models.keys())[0]
    pipe = models[first_h]["mean"]
    preprocess = pipe.named_steps["preprocess"]

    cols = []
    for _, _, transformer_cols in preprocess.transformers_:
        cols.extend(list(transformer_cols))

    # preserve order, remove duplicates
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def choose_horizon(hours_to_peak: float) -> int | None:
    if pd.isna(hours_to_peak):
        return None
    if hours_to_peak <= 0:
        return None
    h = int(math.ceil(hours_to_peak))
    if h < 1 or h > 9:
        return None
    return h


def load_existing(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        for c in expected_cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[expected_cols].copy()
    return pd.DataFrame(columns=expected_cols)


# ---------------------------------------------------------------------
# OPEN-METEO DOWNLOAD
# ---------------------------------------------------------------------
def fetch_open_meteo_airport(meta: dict) -> pd.DataFrame:
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_direction_10m",
        "wind_speed_10m",
        "wind_gusts_10m",
        "surface_pressure",
        "cloud_cover",
        "weather_code",
        "is_day",
    ]

    daily_vars = [
        "sunrise",
        "sunset",
        "temperature_2m_max",
    ]

    params = {
        "latitude": meta["lat"],
        "longitude": meta["lon"],
        "timezone": meta["tz"],
        "hourly": ",".join(hourly_vars),
        "daily": ",".join(daily_vars),
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    r = requests.get(OPEN_METEO_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    if not hourly or "time" not in hourly:
        raise RuntimeError(f"No hourly data returned for {meta['icao']}")

    hdf = pd.DataFrame({
        "timestamp_local": pd.to_datetime(hourly["time"], errors="coerce"),
        "temp": pd.to_numeric(hourly.get("temperature_2m"), errors="coerce"),
        "rhum": pd.to_numeric(hourly.get("relative_humidity_2m"), errors="coerce"),
        "prcp": pd.to_numeric(hourly.get("precipitation"), errors="coerce"),
        "wdir": pd.to_numeric(hourly.get("wind_direction_10m"), errors="coerce"),
        "wspd": pd.to_numeric(hourly.get("wind_speed_10m"), errors="coerce"),
        "wpgt": pd.to_numeric(hourly.get("wind_gusts_10m"), errors="coerce"),
        "pres": pd.to_numeric(hourly.get("surface_pressure"), errors="coerce"),
        "cldc": pd.to_numeric(hourly.get("cloud_cover"), errors="coerce"),
        "coco": pd.to_numeric(hourly.get("weather_code"), errors="coerce"),
        "is_daylight": pd.to_numeric(hourly.get("is_day"), errors="coerce"),
    })

    ddf = pd.DataFrame({
        "date_local": pd.to_datetime(daily.get("time"), errors="coerce").normalize(),
        "sunrise_local": pd.to_datetime(daily.get("sunrise"), errors="coerce"),
        "sunset_local": pd.to_datetime(daily.get("sunset"), errors="coerce"),
        "daily_max_temp": pd.to_numeric(daily.get("temperature_2m_max"), errors="coerce"),
    })

    hdf["date_local"] = hdf["timestamp_local"].dt.normalize()
    out = hdf.merge(ddf, on="date_local", how="left")

    return out.sort_values("timestamp_local").reset_index(drop=True)


# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------
def add_time_and_solar_features(df: pd.DataFrame, meta: dict, peak_offsets: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["airport"] = meta["city"]
    out["city"] = meta["city"]
    out["icao"] = meta["icao"]
    out["latitude"] = meta["lat"]
    out["longitude"] = meta["lon"]
    out["is_us_airport"] = int(meta["is_us_airport"])
    out["is_coastal_airport"] = int(meta["is_coastal_airport"])

    out["month"] = out["timestamp_local"].dt.month
    out["day_of_year"] = out["timestamp_local"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

    out = out.merge(
        peak_offsets.rename(columns={"mean_peak_offset_hours": "peak_offset_hours"}),
        on=["city", "month"],
        how="left",
    )

    out["solar_noon_local"] = out["sunrise_local"] + (out["sunset_local"] - out["sunrise_local"]) / 2
    out["predicted_peak_time_local"] = out["solar_noon_local"] + pd.to_timedelta(out["peak_offset_hours"], unit="h")
    out["hours_from_solar_peak"] = (
        (out["timestamp_local"] - out["predicted_peak_time_local"]).dt.total_seconds() / 3600.0
    )

    out["solar_hour_angle_deg"] = (
        (out["timestamp_local"] - out["solar_noon_local"]).dt.total_seconds() / 3600.0
    ) * 15.0

    lat_rad = np.deg2rad(out["latitude"].astype(float))
    decl_deg = 23.44 * np.sin(2 * np.pi * (284 + out["day_of_year"].astype(float)) / 365.25)
    decl_rad = np.deg2rad(decl_deg)
    hra_rad = np.deg2rad(out["solar_hour_angle_deg"].astype(float))

    sin_elev = (
        np.sin(lat_rad) * np.sin(decl_rad)
        + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hra_rad)
    )
    sin_elev = np.clip(sin_elev, -1.0, 1.0)
    out["solar_elevation_deg"] = np.rad2deg(np.arcsin(sin_elev))

    out["is_daylight"] = out["is_daylight"].fillna(0).astype(float)
    out["is_raining"] = (out["prcp"].fillna(0) > 0).astype(float)

    out["rain_3h_sum"] = out["prcp"].rolling(3, min_periods=1).sum()
    out["rain_6h_sum"] = out["prcp"].rolling(6, min_periods=1).sum()
    out["rain_12h_sum"] = out["prcp"].rolling(12, min_periods=1).sum()
    out["rain_24h_sum"] = out["prcp"].rolling(24, min_periods=1).sum()

    hours_since = []
    last_rain_ts = None
    for ts, raining in zip(out["timestamp_local"], out["is_raining"]):
        if raining == 1:
            last_rain_ts = ts
            hours_since.append(0.0)
        else:
            if last_rain_ts is None:
                hours_since.append(999.0)
            else:
                hours_since.append((ts - last_rain_ts).total_seconds() / 3600.0)
    out["hours_since_rain"] = hours_since

    out["temp_trend_1h"] = out["temp"].diff(1)

    out["solar_heating_efficiency"] = np.where(
        out["solar_elevation_deg"] > 0,
        out["temp"] / (out["solar_elevation_deg"] + 1.0),
        0.0,
    )

    daily_max = (
        out.groupby("date_local", as_index=False)["temp"]
        .max()
        .rename(columns={"temp": "temp_max_that_day"})
        .sort_values("date_local")
    )
    daily_max["yesterday_max_temp"] = daily_max["temp_max_that_day"].shift(1)
    out = out.merge(daily_max[["date_local", "yesterday_max_temp"]], on="date_local", how="left")

    for k in range(1, 25):
        out[f"temp_lag_{k}h"] = out["temp"].shift(k)
        out[f"delta_temp_{k}h"] = out["temp"] - out[f"temp_lag_{k}h"]

    for k in [1, 3, 6]:
        out[f"delta_cldc_{k}h"] = out["cldc"] - out["cldc"].shift(k)
        out[f"delta_pres_{k}h"] = out["pres"] - out["pres"].shift(k)

    for k in [3, 6, 12, 24]:
        out[f"temp_roll_mean_{k}h"] = out["temp"].rolling(k, min_periods=1).mean()
        out[f"temp_roll_std_{k}h"] = out["temp"].rolling(k, min_periods=2).std()

    return out


def keep_today_local_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_local"] = pd.to_datetime(out["timestamp_local"], errors="coerce").dt.normalize()
    latest_date = out.groupby("airport", as_index=False)["date_local"].max().rename(columns={"date_local": "keep_date"})
    out = out.merge(latest_date, on="airport", how="left")
    out = out[out["date_local"] == out["keep_date"]].copy()
    return out.drop(columns=["keep_date"])


# ---------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------
def score_latest_rows(obs_model_today: pd.DataFrame, models: dict, required_model_cols: list[str]) -> pd.DataFrame:
    rows = []

    for airport, x in obs_model_today.groupby("airport", sort=True):
        x = x.sort_values("timestamp_local").copy()
        if x.empty:
            continue

        latest = x.iloc[-1].copy()

        hours_to_peak = -safe_float(latest.get("hours_from_solar_peak"))
        model_bucket_num = choose_horizon(hours_to_peak)
        if model_bucket_num is None:
            print(f"{airport}: skipped scoring (hours_to_peak={hours_to_peak})")
            continue

        if model_bucket_num not in models or "mean" not in models[model_bucket_num]:
            print(f"{airport}: missing mean model for h{model_bucket_num}")
            continue

        feature_row = pd.DataFrame([latest])

        for c in required_model_cols:
            if c not in feature_row.columns:
                feature_row[c] = np.nan

        feature_row = feature_row[required_model_cols].copy()

        score_row = {
            "airport": airport,
            "run_timestamp_local": latest["timestamp_local"],
            "model_bucket": f"h{model_bucket_num}",
            "hours_to_peak": hours_to_peak,
            "solar_noon_local": latest.get("solar_noon_local"),
            "peak_offset_hours": safe_float(latest.get("peak_offset_hours")),
            "predicted_peak_time_local": latest.get("predicted_peak_time_local"),
            "projected_max_temp": np.nan,
            "projected_q50": np.nan,
            "projected_q90": np.nan,
            "projected_q95": np.nan,
        }

        score_row["projected_max_temp"] = safe_float(
            models[model_bucket_num]["mean"].predict(feature_row)[0]
        )

        for q in ["q50", "q90", "q95"]:
            if q in models[model_bucket_num]:
                score_row[f"projected_{q}"] = safe_float(
                    models[model_bucket_num][q].predict(feature_row)[0]
                )

        rows.append(score_row)
        print(
            f"{airport}: h{model_bucket_num} "
            f"mean={score_row['projected_max_temp']:.3f} "
            f"q50={score_row['projected_q50']:.3f} "
            f"q90={score_row['projected_q90']:.3f} "
            f"q95={score_row['projected_q95']:.3f}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    peak_offsets = load_peak_offsets(PEAK_OFFSET_PATH)
    models = discover_models(MODEL_DIR)
    required_model_cols = get_required_columns_from_models(models)

    print("Loaded model horizons:", sorted(models.keys()))
    print(f"Required model columns: {len(required_model_cols)}")

    missing_horizons = [h for h in range(1, 10) if h not in models]
    if missing_horizons:
        print(f"WARNING: missing horizons: {missing_horizons}")

    for h in sorted(models.keys()):
        missing_parts = [k for k in ["mean", "q50", "q90", "q95"] if k not in models[h]]
        if missing_parts:
            print(f"WARNING: h{h} missing parts: {missing_parts}")

    all_obs_rows = []

    for airport, meta in AIRPORTS.items():
        print(f"Downloading latest data for {airport} ({meta['icao']})")
        try:
            raw = fetch_open_meteo_airport(meta)
            feat = add_time_and_solar_features(raw, meta, peak_offsets)
            all_obs_rows.append(feat)
            print(f"  rows={len(feat)} latest_ts={feat['timestamp_local'].max()}")
        except Exception as e:
            print(f"  FAILED: {e}")

    if not all_obs_rows:
        print("No airport data downloaded. Exiting.")
        return

    obs_model_full = pd.concat(all_obs_rows, ignore_index=True)
    obs_model_today = keep_today_local_only(obs_model_full).copy()

    obs_model_today = (
        obs_model_today
        .sort_values(["airport", "timestamp_local"])
        .reset_index(drop=True)
    )
    obs_model_today.to_parquet(OBS_MODEL_PATH, index=False)
    print(f"Saved observations model rows: {len(obs_model_today):,} -> {OBS_MODEL_PATH}")

    scored = score_latest_rows(obs_model_today, models, required_model_cols)
 
    forecast_cols = [
        "airport",
        "run_timestamp_local",
        "model_bucket",
        "hours_to_peak",
        "solar_noon_local",
        "peak_offset_hours",
        "predicted_peak_time_local",
        "projected_max_temp",
        "projected_q50",
        "projected_q90",
        "projected_q95",
    ]
    
    if scored.empty:
        print("No forecast model rows scored. Writing empty forecast_model.parquet.")
        if not FORECAST_MODEL_PATH.exists():
            pd.DataFrame(columns=forecast_cols).to_parquet(FORECAST_MODEL_PATH, index=False)
        return

    forecast_cols = [
        "airport",
        "run_timestamp_local",
        "model_bucket",
        "hours_to_peak",
        "solar_noon_local",
        "peak_offset_hours",
        "predicted_peak_time_local",
        "projected_max_temp",
        "projected_q50",
        "projected_q90",
        "projected_q95",
    ]

    existing = load_existing(FORECAST_MODEL_PATH, forecast_cols)
    combined = pd.concat([existing, scored[forecast_cols]], ignore_index=True)

    combined["run_timestamp_local"] = pd.to_datetime(combined["run_timestamp_local"], errors="coerce")
    for c in ["hours_to_peak", "peak_offset_hours", "projected_max_temp", "projected_q50", "projected_q90", "projected_q95"]:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")

    combined = combined.dropna(subset=["airport", "run_timestamp_local"]).copy()

    combined["run_hour_local"] = combined["run_timestamp_local"].dt.floor("1h")
    combined = (
        combined.sort_values(["airport", "run_timestamp_local"])
        .drop_duplicates(subset=["airport", "run_hour_local"], keep="last")
        .drop(columns=["run_hour_local"])
        .sort_values(["airport", "run_timestamp_local"])
        .reset_index(drop=True)
    )

    combined.to_parquet(FORECAST_MODEL_PATH, index=False)
    print(f"Saved forecast model rows: {len(combined):,} -> {FORECAST_MODEL_PATH}")


if __name__ == "__main__":
    main()
