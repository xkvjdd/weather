import io
import os
import pandas as pd
import requests

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

CITY_TO_TZ = {
    "ATL": "America/New_York",
    "NYC": "America/New_York",
    "CHI": "America/Chicago",
    "DAL": "America/Chicago",
    "SEA": "America/Los_Angeles",
    "MIA": "America/New_York",
    "TOR": "America/Toronto",
    "PAR": "Europe/Paris",
    "SEL": "Asia/Seoul",
    "ANK": "Europe/Istanbul",
    "BUE": "America/Argentina/Buenos_Aires",
    "LON": "Europe/London",
    "WLG": "Pacific/Auckland",
}

BASE_URL = "https://data.meteostat.net/hourly/{year}/{station}.csv.gz"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python requests"
TIMEOUT = 60

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "observations.parquet")

NOW = pd.Timestamp.now("UTC")
YEAR = NOW.year


def download_station_year(station: str, year: int) -> pd.DataFrame:
    url = BASE_URL.format(year=year, station=station)
    resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})

    if resp.status_code == 404:
        return pd.DataFrame()

    resp.raise_for_status()
    return pd.read_csv(io.BytesIO(resp.content), compression="gzip")


def parse_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    lower_cols = {str(c).lower(): c for c in out.columns}

    if "time" in lower_cols:
        out["timestamp_utc"] = pd.to_datetime(out[lower_cols["time"]], errors="coerce", utc=True)

    elif "date" in lower_cols and "hour" in lower_cols:
        date_col = lower_cols["date"]
        hour_col = lower_cols["hour"]
        out["timestamp_utc"] = pd.to_datetime(
            out[date_col].astype(str) + " " + out[hour_col].astype(str).str.zfill(2) + ":00:00",
            errors="coerce",
            utc=True,
        )

    else:
        first_col = out.columns[0]
        out["timestamp_utc"] = pd.to_datetime(out[first_col], errors="coerce", utc=True)

    out = out[out["timestamp_utc"].notna()].copy()
    return out


frames = []

for airport, meta in STATIONS.items():
    station = meta["meteostat_id"]
    print(f"Downloading {airport}")

    try:
        df = download_station_year(station, YEAR)
        if df.empty:
            print(f"  no file for {YEAR}")
            continue

        df = parse_timestamp_column(df)
        if df.empty:
            print("  empty after timestamp parse")
            continue

        if "temp" not in df.columns:
            print(f"  missing temp column; cols={list(df.columns)}")
            continue

        max_ts = df["timestamp_utc"].max()
        min_ts = df["timestamp_utc"].min()
        cut = max_ts - pd.Timedelta(hours=96)

        print(f"  source range: {min_ts} -> {max_ts}")
        print(f"  using cut:    {cut}")

        df = df[df["timestamp_utc"] >= cut].copy()
        if df.empty:
            print("  no rows after source-relative 96h cut")
            continue

        tz_name = CITY_TO_TZ[airport]

        out = pd.DataFrame({
            "airport": airport,
            "timestamp_utc": df["timestamp_utc"].dt.tz_localize(None),
            "timestamp_local": df["timestamp_utc"].dt.tz_convert(tz_name).dt.tz_localize(None),
            "temp": pd.to_numeric(df["temp"], errors="coerce"),
        })

        out = out[out["temp"].notna()].sort_values("timestamp_local").reset_index(drop=True)
        frames.append(out)
        print(f"  kept {len(out):,} rows")

    except Exception as e:
        print(f"  FAILED: {e}")

if frames:
    final = pd.concat(frames, ignore_index=True)
    final = final.sort_values(["airport", "timestamp_local"]).reset_index(drop=True)
else:
    final = pd.DataFrame(columns=["airport", "timestamp_utc", "timestamp_local", "temp"])

final.to_parquet(OUTPUT_PATH, index=False)

print(f"\nSaved {len(final):,} rows -> {OUTPUT_PATH}")
