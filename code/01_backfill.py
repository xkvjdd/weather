"""
fetch_observations.py
─────────────────────
Fetches the last 3 days of hourly temperature observations for each airport
using the Open-Meteo Historical Forecast API (free, no API key required).

Data is already in local time (Open-Meteo returns local timestamps when
timezone is specified).

Output: data/dashboard/observations.parquet
Columns: airport | timestamp_utc | timestamp_local | temp
"""

import os
import time
import requests
import pandas as pd

# ── Airport config ────────────────────────────────────────────────────────────
# lat/lon sourced from standard airport coordinates
STATIONS = {
    "ATL": {"lat": 33.6407,  "lon": -84.4277,  "tz": "America/New_York"},
    "NYC": {"lat": 40.7772,  "lon": -73.8726,  "tz": "America/New_York"},
    "CHI": {"lat": 41.9742,  "lon": -87.9073,  "tz": "America/Chicago"},
    "DAL": {"lat": 32.8481,  "lon": -96.8512,  "tz": "America/Chicago"},
    "SEA": {"lat": 47.4502,  "lon": -122.3088, "tz": "America/Los_Angeles"},
    "MIA": {"lat": 25.7959,  "lon": -80.2870,  "tz": "America/New_York"},
    "TOR": {"lat": 43.6777,  "lon": -79.6248,  "tz": "America/Toronto"},
    "PAR": {"lat": 49.0097,  "lon":   2.5479,  "tz": "Europe/Paris"},
    "SEL": {"lat": 37.4691,  "lon": 126.4510,  "tz": "Asia/Seoul"},
    "ANK": {"lat": 40.1281,  "lon":  32.9951,  "tz": "Europe/Istanbul"},
    "BUE": {"lat": -34.8222, "lon": -58.5358,  "tz": "America/Argentina/Buenos_Aires"},
    "LON": {"lat": 51.4775,  "lon":  -0.4614,  "tz": "Europe/London"},
    "WLG": {"lat": -41.3272, "lon": 174.8052,  "tz": "Pacific/Auckland"},
}

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(DATA_DIR, "observations.parquet")

# ── Date range: today and previous 2 days (3 days total) ─────────────────────
NOW       = pd.Timestamp.now("UTC")
END_DATE  = NOW.strftime("%Y-%m-%d")
START_DATE = (NOW - pd.Timedelta(days=2)).strftime("%Y-%m-%d")

BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

TIMEOUT = 30
RETRY_WAIT = 5   # seconds between retries on rate-limit


def fetch_airport(airport: str, meta: dict) -> pd.DataFrame:
    params = {
        "latitude":        meta["lat"],
        "longitude":       meta["lon"],
        "start_date":      START_DATE,
        "end_date":        END_DATE,
        "hourly":          "temperature_2m",
        "timezone":        meta["tz"],
        "temperature_unit": "celsius",
    }

    for attempt in range(3):
        resp = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
        if resp.status_code == 429:
            print(f"  rate-limited, waiting {RETRY_WAIT}s …")
            time.sleep(RETRY_WAIT)
            continue
        resp.raise_for_status()
        break

    data = resp.json()

    hourly  = data.get("hourly", {})
    times   = hourly.get("time", [])
    temps   = hourly.get("temperature_2m", [])

    if not times:
        print(f"  no hourly data returned")
        return pd.DataFrame()

    df = pd.DataFrame({"timestamp_local_str": times, "temp": temps})
    df = df[df["temp"].notna()].copy()

    # Parse local timestamp (Open-Meteo returns ISO strings in the requested tz)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local_str"])

    # Derive UTC
    local_tz = meta["tz"]
    df["timestamp_utc"] = (
        df["timestamp_local"]
        .dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    df["timestamp_local"] = df["timestamp_local"].dt.tz_localize(None)
    df["airport"] = airport

    return df[["airport", "timestamp_utc", "timestamp_local", "temp"]].sort_values("timestamp_local").reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────
frames = []

for airport, meta in STATIONS.items():
    print(f"Fetching {airport} ({meta['tz']})  {START_DATE} → {END_DATE}")
    try:
        df = fetch_airport(airport, meta)
        if df.empty:
            print(f"  ✗ no data")
        else:
            print(f"  ✓ {len(df):,} rows  (latest: {df['timestamp_local'].max()})")
            frames.append(df)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

    time.sleep(0.5)   # polite pacing between airports

if frames:
    final = pd.concat(frames, ignore_index=True)
    final = final.sort_values(["airport", "timestamp_local"]).reset_index(drop=True)
else:
    final = pd.DataFrame(columns=["airport", "timestamp_utc", "timestamp_local", "temp"])

final.to_parquet(OUTPUT_PATH, index=False)
print(f"\n✓ Saved {len(final):,} rows → {OUTPUT_PATH}")
print(f"  Airports: {sorted(final['airport'].unique().tolist())}")
print(f"  Time range: {final['timestamp_local'].min()} → {final['timestamp_local'].max()}")
