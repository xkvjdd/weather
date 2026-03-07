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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "forecast_latest.parquet")

HTTP_TIMEOUT = 30
SLEEP_SECONDS = 0.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    )
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
# Source 1: Open-Meteo best-match daily max — today only
# ---------------------------------------------------------------------------

def fetch_open_meteo_daily_max(
    session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
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

    r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    dates = data.get("daily", {}).get("time", [])
    vals = data.get("daily", {}).get("temperature_2m_max", [])

    for d, v in zip(dates, vals):
        if d == today and v is not None:
            return clean_temp_to_celsius(v, "C")

    print(f"    WARNING: open-meteo daily — no match for {today}, got dates: {dates}")
    return None


# ---------------------------------------------------------------------------
# Source 2: Open-Meteo GFS hourly max — today only, strict date filter
# ---------------------------------------------------------------------------

def fetch_open_meteo_gfs_hourly_max(
    session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
) -> float | None:
    url = "https://api.open-meteo.com/v1/gfs"
    today = today_local_str(tz_name)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "start_date": today,
        "end_date": today,
        "timezone": tz_name,
        "temperature_unit": "celsius",
    }

    r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    times = data.get("hourly", {}).get("time", [])
    temps = data.get("hourly", {}).get("temperature_2m", [])

    xs = []
    for t, temp in zip(times, temps):
        if str(t)[:10] == today and temp is not None:
            v = clean_temp_to_celsius(temp, "C")
            if v is not None:
                xs.append(v)

    if xs:
        return round(max(xs), 3)

    print(f"    WARNING: GFS hourly — no hours matched {today}, available: {sorted(set(str(t)[:10] for t in times))}")
    return None


# ---------------------------------------------------------------------------
# Source 3: Open-Meteo historical/analysis max — today only
# This uses the historical forecast API which returns analysed values for
# today rather than a forward forecast, giving a grounded observed-ish high.
# ---------------------------------------------------------------------------

def fetch_open_meteo_historical_max(
    session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
) -> float | None:
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    today = today_local_str(tz_name)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": today,
        "end_date": today,
        "daily": "temperature_2m_max",
        "timezone": tz_name,
        "temperature_unit": "celsius",
    }

    r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    dates = data.get("daily", {}).get("time", [])
    vals = data.get("daily", {}).get("temperature_2m_max", [])

    for d, v in zip(dates, vals):
        if d == today and v is not None:
            return clean_temp_to_celsius(v, "C")

    print(f"    WARNING: historical forecast API — no match for {today}, got dates: {dates}")
    return None


# ---------------------------------------------------------------------------
# Per-airport fetch
# ---------------------------------------------------------------------------

def fetch_airport_forecasts(session: requests.Session, airport: str, meta: dict) -> dict:
    lat = meta["lat"]
    lon = meta["lon"]
    tz_name = meta["tz"]
    icao = meta["icao"]

    pulled_at_local = now_local_naive(tz_name)
    forecast_date_local = today_local_str(tz_name)

    src1 = src2 = src3 = None

    try:
        src1 = fetch_open_meteo_daily_max(session, lat, lon, tz_name)
    except Exception as e:
        print(f"  S1 open-meteo daily FAILED: {e}")

    try:
        src2 = fetch_open_meteo_gfs_hourly_max(session, lat, lon, tz_name)
    except Exception as e:
        print(f"  S2 GFS hourly FAILED: {e}")

    try:
        src3 = fetch_open_meteo_historical_max(session, lat, lon, tz_name)
    except Exception as e:
        print(f"  S3 historical forecast FAILED: {e}")

    avg_val = mean_ignore_none([src1, src2, src3])

    return {
        "airport": airport,
        "icao": icao,
        "forecast_date_local": forecast_date_local,
        "pulled_at_local": pulled_at_local,
        "forecast_source_1": src1,
        "forecast_source_2": src2,
        "forecast_source_3": src3,
        "forecast_avg_max": avg_val,
        "source_1_name": "open_meteo_best_match_daily_max",
        "source_2_name": "open_meteo_gfs_hourly_day_max",
        "source_3_name": "open_meteo_historical_forecast_daily_max",
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
        print(f"Fetching forecast for {airport} ({meta['icao']}) — local date: {today_local_str(meta['tz'])}")
        try:
            row = fetch_airport_forecasts(session, airport, meta)
            rows.append(row)
            print(
                f"  S1={row['forecast_source_1']} "
                f"S2={row['forecast_source_2']} "
                f"S3={row['forecast_source_3']} "
                f"AVG={row['forecast_avg_max']}"
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
