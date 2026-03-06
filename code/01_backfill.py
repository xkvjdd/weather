import io
import os
import pandas as pd
import requests
from datetime import datetime, timedelta

# same stations as your 01_data.py
STATIONS = {
    "ATL": {"icao": "KATL", "meteostat_id": "72219"},
    "NYC": {"icao": "KLGA", "meteostat_id": "72503"},
    "CHI": {"icao": "KORD", "meteostat_id": "72530"},
    "DAL": {"icao": "KDAL", "meteostat_id": "72258"},
    "SEA": {"icao": "KSEA", "meteostat_id": "72793"},
    "MIA": {"icao": "KMIA", "meteostat_id": "72202"},
    "TOR": {"icao": "CYYZ", "meteostat_id": "71624"},
    "PAR": {"icao": "LFPG", "meteostat_id": "07157"},
    "SEL": {"icao": "RKSI", "meteostat_id": "47113"},
    "ANK": {"icao": "LTAC", "meteostat_id": "17128"},
    "BUE": {"icao": "SAEZ", "meteostat_id": "87576"},
    "LON": {"icao": "EGLL", "meteostat_id": "03772"},
    "WLG": {"icao": "NZWN", "meteostat_id": "93436"},
}

BASE_URL = "https://data.meteostat.net/hourly/{year}/{station}.csv.gz"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "observations.parquet")

NOW = pd.Timestamp.utcnow()
CUT = NOW - pd.Timedelta(hours=96)


def download_station_year(station, year):

    url = BASE_URL.format(year=year, station=station)

    r = requests.get(url, timeout=60)

    if r.status_code == 404:
        return pd.DataFrame()

    r.raise_for_status()

    return pd.read_csv(io.BytesIO(r.content), compression="gzip")


frames = []

year = datetime.utcnow().year

for airport, meta in STATIONS.items():

    station = meta["meteostat_id"]

    print(f"Downloading {airport}")

    df = download_station_year(station, year)

    if df.empty:
        continue

    df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")

    df = df[df["timestamp"] >= CUT]

    if df.empty:
        continue

    out = pd.DataFrame(
        {
            "airport": airport,
            "timestamp_local": df["timestamp"],
            "temp": df["temp"],
        }
    )

    frames.append(out)


if frames:

    final = pd.concat(frames, ignore_index=True)

else:

    final = pd.DataFrame(columns=["airport", "timestamp_local", "temp"])


final.to_parquet(OUTPUT_PATH, index=False)

print(f"\nSaved {len(final):,} rows")
print(OUTPUT_PATH)