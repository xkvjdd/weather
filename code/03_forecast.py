import os
import re
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


def f_to_c(x: float) -> float:
    return (x - 32.0) * 5.0 / 9.0


def clean_temp_to_celsius(value, unit: str | None = None) -> float | None:
    if value is None:
        return None

    try:
        x = float(value)
    except Exception:
        return None

    u = (unit or "C").upper()
    if u == "F":
        return round(f_to_c(x), 3)
    return round(x, 3)


def parse_high_from_wunderground_html(html: str) -> float | None:
    if not html:
        return None

    patterns = [
        r"High\s+(-?\d+(?:\.\d+)?)\s*°?\s*([CF])",
        r"High\s+(-?\d+(?:\.\d+)?)\s*([CF])",
        r"High\s+(-?\d+(?:\.\d+)?)\b",
    ]

    for pat in patterns:
        m = re.search(pat, html, flags=re.IGNORECASE)
        if not m:
            continue

        value = m.group(1)
        unit = m.group(2) if len(m.groups()) >= 2 else "F"
        return clean_temp_to_celsius(value, unit)

    return None


def fetch_wunderground_forecast_max(session: requests.Session, icao: str) -> float | None:
    url = f"https://www.wunderground.com/hourly/{icao}"
    r = session.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return parse_high_from_wunderground_html(r.text)


def fetch_open_meteo_daily_max(
    session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
) -> float | None:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "forecast_days": 2,
        "timezone": tz_name,
        "temperature_unit": "celsius",
    }

    r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    vals = daily.get("temperature_2m_max", [])

    if not dates or not vals:
        return None

    target_day = today_local_str(tz_name)
    for d, v in zip(dates, vals):
        if d == target_day:
            return clean_temp_to_celsius(v, "C")

    return clean_temp_to_celsius(vals[0], "C")


def fetch_open_meteo_gfs_hourly_max(
    session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
) -> float | None:
    url = "https://api.open-meteo.com/v1/gfs"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "forecast_days": 2,
        "timezone": tz_name,
        "temperature_unit": "celsius",
    }

    r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    if not times or not temps:
        return None

    target_day = today_local_str(tz_name)

    xs = []
    for t, temp in zip(times, temps):
        try:
            d = str(t)[:10]
        except Exception:
            continue
        if d == target_day:
            y = clean_temp_to_celsius(temp, "C")
            if y is not None:
                xs.append(y)

    if xs:
        return round(max(xs), 3)

    ys = [clean_temp_to_celsius(v, "C") for v in temps]
    ys = [v for v in ys if v is not None]
    if not ys:
        return None
    return round(max(ys), 3)


def mean_ignore_none(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return round(sum(xs) / len(xs), 3)


def fetch_airport_forecasts(session: requests.Session, airport: str, meta: dict) -> dict:
    icao = meta["icao"]
    lat = meta["lat"]
    lon = meta["lon"]
    tz_name = meta["tz"]

    pulled_at_local = now_local_naive(tz_name)
    forecast_date_local = today_local_str(tz_name)

    src1 = src2 = src3 = None

    try:
        src1 = fetch_wunderground_forecast_max(session, icao)
    except Exception as e:
        print(f"  S1 Wunderground FAILED: {e}")

    try:
        src2 = fetch_open_meteo_daily_max(session, lat, lon, tz_name)
    except Exception as e:
        print(f"  S2 Open-Meteo FAILED: {e}")

    try:
        src3 = fetch_open_meteo_gfs_hourly_max(session, lat, lon, tz_name)
    except Exception as e:
        print(f"  S3 GFS FAILED: {e}")

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
        "source_1_name": "wunderground_hourly_high",
        "source_2_name": "open_meteo_best_match_daily_max",
        "source_3_name": "open_meteo_gfs_hourly_day_max",
    }


def load_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_parquet(path)
        keep = [
            c for c in [
                "airport",
                "icao",
                "forecast_date_local",
                "pulled_at_local",
                "forecast_source_1",
                "forecast_source_2",
                "forecast_source_3",
                "forecast_avg_max",
                "source_1_name",
                "source_2_name",
                "source_3_name",
            ]
            if c in df.columns
        ]
        if keep:
            return df[keep].copy()

    return pd.DataFrame(
        columns=[
            "airport",
            "icao",
            "forecast_date_local",
            "pulled_at_local",
            "forecast_source_1",
            "forecast_source_2",
            "forecast_source_3",
            "forecast_avg_max",
            "source_1_name",
            "source_2_name",
            "source_3_name",
        ]
    )


def main() -> None:
    session = make_session()
    rows = []

    for airport, meta in AIRPORTS.items():
        print(f"Fetching forecast for {airport} ({meta['icao']})")
        try:
            row = fetch_airport_forecasts(session, airport, meta)
            rows.append(row)
            print(
                "  "
                f"S1={row['forecast_source_1']} "
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
    for col in [
        "forecast_source_1",
        "forecast_source_2",
        "forecast_source_3",
        "forecast_avg_max",
    ]:
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
