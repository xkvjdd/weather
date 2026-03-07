import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

AIRPORTS = {
    "ATL": {"icao": "KATL", "lat": 33.6407, "lon": -84.4277, "tz": "America/New_York"},
    "NYC": {"icao": "KLGA", "lat": 40.7769, "lon": -73.8740, "tz": "America/New_York"},
    "CHI": {"icao": "KORD", "lat": 41.9742, "lon": -87.9073, "tz": "America/Chicago"},
    "DAL": {"icao": "KDAL", "lat": 32.8471, "lon": -96.8517, "tz": "America/Chicago"},
    "SEA": {"icao": "KSEA", "lat": 47.4502, "lon": -122.3088, "tz": "America/Los_Angeles"},
    "MIA": {"icao": "KMIA", "lat": 25.7959, "lon": -80.2870, "tz": "America/New_York"},
    "TOR": {"icao": "CYYZ", "lat": 43.6777, "lon": -79.6248, "tz": "America/Toronto"},
    "PAR": {"icao": "LFPG", "lat": 49.0097, "lon": 2.5479, "tz": "Europe/Paris"},
    "SEL": {"icao": "RKSI", "lat": 37.4602, "lon": 126.4407, "tz": "Asia/Seoul"},
    "ANK": {"icao": "LTAC", "lat": 40.1281, "lon": 32.9951, "tz": "Europe/Istanbul"},
    "BUE": {"icao": "SAEZ", "lat": -34.8222, "lon": -58.5358, "tz": "America/Argentina/Buenos_Aires"},
    "LON": {"icao": "EGLL", "lat": 51.4700, "lon": -0.4543, "tz": "Europe/London"},
    "WLG": {"icao": "NZWN", "lat": -41.3272, "lon": 174.8050, "tz": "Pacific/Auckland"},
}

# US airports that can use NOAA NWS API
US_AIRPORTS = {"ATL", "NYC", "CHI", "DAL", "SEA", "MIA"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "forecast_latest.parquet")

HTTP_TIMEOUT = 30
SLEEP_SECONDS = 0.5

HEADERS = {
    "User-Agent": "airport-weather-dashboard/1.0 (research project)"
}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def now_local_naive(tz_name: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.now(ZoneInfo(tz_name)).replace(tzinfo=None))


def today_local_str(tz_name: str) -> str:
    return datetime.now(ZoneInfo(tz_name)).date().isoformat()


def clean_temp_to_celsius(value, unit: str | None = None) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    u = (unit or "C").upper()
    if u == "F":
        return round((x - 32.0) * 5.0 / 9.0, 3)
    return round(x, 3)


def mean_ignore_none(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    result = round(sum(xs) / len(xs), 3)
    if len(xs) > 1 and (max(xs) - min(xs)) > 3.0:
        print(f"    WARNING: Source spread > 3°C: {xs}")
    return result


# ---------------------------------------------------------------------------
# Source: NOAA NWS (US airports only)
# Official US government forecast — completely independent model
# ---------------------------------------------------------------------------

def fetch_noaa_today_high(
    session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
) -> float | None:
    # Step 1: resolve lat/lon to NWS grid point
    r = session.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    forecast_url = r.json()["properties"]["forecast"]

    # Step 2: fetch gridpoint forecast periods
    r2 = session.get(forecast_url, timeout=HTTP_TIMEOUT)
    r2.raise_for_status()
    periods = r2.json()["properties"]["periods"]

    today = today_local_str(tz_name)
    for period in periods:
        if period["startTime"][:10] == today and period["isDaytime"]:
            return clean_temp_to_celsius(period["temperature"], period.get("temperatureUnit", "F"))

    print(f"    WARNING: NOAA — no daytime period for {today}")
    return None


# ---------------------------------------------------------------------------
# Source: Open-Meteo with explicit model selection
# Passing start_date=end_date=today (local) ensures no cross-day bleed
# ---------------------------------------------------------------------------

def fetch_open_meteo_daily_max(
    session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
    model: str | None = None,
) -> float | None:
    url = "https://api.open-meteo.com/v1/forecast"
    today = today_local_str(tz_name)
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "start_date": today,
        "end_date": today,
        "timezone": tz_name,
        "temperature_unit": "celsius",
    }
    if model:
        params["models"] = model

    r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    dates = data.get("daily", {}).get("time", [])
    vals = data.get("daily", {}).get("temperature_2m_max", [])

    for d, v in zip(dates, vals):
        if d == today and v is not None:
            return clean_temp_to_celsius(v, "C")

    print(f"    WARNING: Open-Meteo [{model or 'best_match'}] — no match for {today}, got: {dates}")
    return None


# ---------------------------------------------------------------------------
# Per-airport fetch — routes to different source combos by region
#
# US airports  → NOAA (independent) + GFS + ECMWF  (3 truly different models)
# Intl airports → GFS + ECMWF + best-match blend    (2 different NWP + ensemble)
# ---------------------------------------------------------------------------

def fetch_airport_forecasts(session: requests.Session, airport: str, meta: dict) -> dict:
    lat, lon, tz_name, icao = meta["lat"], meta["lon"], meta["tz"], meta["icao"]
    is_us = airport in US_AIRPORTS

    pulled_at_local = now_local_naive(tz_name)
    forecast_date_local = today_local_str(tz_name)

    src1 = src2 = src3 = None

    if is_us:
        s1_name = "noaa_nws_daytime_high"
        s2_name = "open_meteo_gfs_seamless_daily_max"
        s3_name = "open_meteo_ecmwf_ifs04_daily_max"

        try:
            src1 = fetch_noaa_today_high(session, lat, lon, tz_name)
        except Exception as e:
            print(f"  S1 NOAA FAILED: {e}")

        try:
            src2 = fetch_open_meteo_daily_max(session, lat, lon, tz_name, model="gfs_seamless")
        except Exception as e:
            print(f"  S2 GFS FAILED: {e}")

        try:
            src3 = fetch_open_meteo_daily_max(session, lat, lon, tz_name, model="ecmwf_ifs04")
        except Exception as e:
            print(f"  S3 ECMWF FAILED: {e}")

    else:
        s1_name = "open_meteo_gfs_seamless_daily_max"
        s2_name = "open_meteo_ecmwf_ifs04_daily_max"
        s3_name = "open_meteo_best_match_daily_max"

        try:
            src1 = fetch_open_meteo_daily_max(session, lat, lon, tz_name, model="gfs_seamless")
        except Exception as e:
            print(f"  S1 GFS FAILED: {e}")

        try:
            src2 = fetch_open_meteo_daily_max(session, lat, lon, tz_name, model="ecmwf_ifs04")
        except Exception as e:
            print(f"  S2 ECMWF FAILED: {e}")

        try:
            src3 = fetch_open_meteo_daily_max(session, lat, lon, tz_name, model=None)
        except Exception as e:
            print(f"  S3 best-match FAILED: {e}")

    return {
        "airport": airport,
        "icao": icao,
        "forecast_date_local": forecast_date_local,
        "pulled_at_local": pulled_at_local,
        "forecast_source_1": src1,
        "forecast_source_2": src2,
        "forecast_source_3": src3,
        "forecast_avg_max": mean_ignore_none([src1, src2, src3]),
        "source_1_name": s1_name,
        "source_2_name": s2_name,
        "source_3_name": s3_name,
    }


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def load_existing(path: str) -> pd.DataFrame:
    cols = [
        "airport", "icao", "forecast_date_local", "pulled_at_local",
        "forecast_source_1", "forecast_source_2", "forecast_source_3",
        "forecast_avg_max", "source_1_name", "source_2_name", "source_3_name",
    ]
    if os.path.exists(path):
        df = pd.read_parquet(path)
        keep = [c for c in cols if c in df.columns]
        if keep:
            return df[keep].copy()
    return pd.DataFrame(columns=cols)


def main() -> None:
    session = make_session()
    rows = []

    for airport, meta in AIRPORTS.items():
        is_us = airport in US_AIRPORTS
        print(
            f"Fetching {airport} ({meta['icao']}) "
            f"[{'US: NOAA+GFS+ECMWF' if is_us else 'INTL: GFS+ECMWF+blend'}] "
            f"local date: {today_local_str(meta['tz'])}"
        )
        try:
            row = fetch_airport_forecasts(session, airport, meta)
            rows.append(row)
            print(
                f"  S1={row['forecast_source_1']} ({row['source_1_name']})\n"
                f"  S2={row['forecast_source_2']} ({row['source_2_name']})\n"
                f"  S3={row['forecast_source_3']} ({row['source_3_name']})\n"
                f"  AVG={row['forecast_avg_max']}"
            )
        except Exception as e:
            print(f"  FAILED: {e}")

        time.sleep(SLEEP_SECONDS)

    if not rows:
        print("No forecast rows fetched; nothing saved.")
        return

    new_df = pd.DataFrame(rows)
    existing = load_existing(OUTPUT_PATH)
    combined = pd.concat([existing, new_df], ignore_index=True)

    combined["pulled_at_local"] = pd.to_datetime(combined["pulled_at_local"], errors="coerce")
    for col in ["forecast_source_1", "forecast_source_2", "forecast_source_3", "forecast_avg_max"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.dropna(subset=["airport", "pulled_at_local"]).copy()

    combined["bucket_5m"] = combined["pulled_at_local"].dt.floor("5min")
    combined = (
        combined.sort_values(["airport", "pulled_at_local"])
        .drop_duplicates(subset=["airport", "bucket_5m"], keep="last")
        .drop(columns=["bucket_5m"])
        .sort_values(["airport", "pulled_at_local"])
        .reset_index(drop=True)
    )

    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(combined):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
