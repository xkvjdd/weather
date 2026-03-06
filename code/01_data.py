import io
import os

import pandas as pd
import requests

# -------------------------------------------------------------------
# Polymarket airport stations mapped to Meteostat station IDs
#
# Source references used for these station IDs / bulk format:
# - Meteostat bulk hourly endpoint:
#   https://dev.meteostat.net/data/timeseries/hourly
# - Station pages on meteostat.net showing Meteostat ID <-> ICAO mapping:
#   KLGA=72503, KATL=72219, KORD=72530, KDAL=72258, KSEA=72793,
#   KMIA=72202, CYYZ=71624, LFPG=07157, RKSI=47113, LTAC=17128,
#   SAEZ=87576, EGLL=03772
# -------------------------------------------------------------------
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

START_YEAR = 2005
END_YEAR = 2026
CUT_OFF_INCLUSIVE = pd.Timestamp("2026-03-01 23:59:59")
RUN_DATE = "2026-03-01"

BASE_URL = "https://data.meteostat.net/hourly/{year}/{station}.csv.gz"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python requests"
TIMEOUT = 60

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "Data")
RUN_FOLDER = os.path.join(DATA_ROOT, RUN_DATE)

os.makedirs(RUN_FOLDER, exist_ok=True)


def download_station_year(meteostat_id: str, year: int) -> pd.DataFrame:
    """
    Download one station-year bulk hourly file from Meteostat.
    Returns empty DataFrame if the file doesn't exist.
    """
    url = BASE_URL.format(year=year, station=meteostat_id)
    resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})

    if resp.status_code == 404:
        return pd.DataFrame()

    resp.raise_for_status()
    return pd.read_csv(io.BytesIO(resp.content), compression="gzip")


def normalize_hourly_df(df: pd.DataFrame, city: str, icao: str, meteostat_id: str) -> pd.DataFrame:
    if df.empty:
        return df

    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        first_col = df.columns[0]
        df["timestamp"] = pd.to_datetime(df[first_col], errors="coerce")

    df = df[df["timestamp"].notna()].copy()

    df["city"] = city
    df["station"] = icao
    df["meteostat_id"] = meteostat_id

    first_cols = ["city", "station", "meteostat_id", "timestamp"]
    remaining = [c for c in df.columns if c not in first_cols]
    return df[first_cols + remaining]


def main() -> None:
    failures = []

    for city, meta in STATIONS.items():
        icao = meta["icao"]
        meteostat_id = meta["meteostat_id"]

        print(f"\nDownloading {city} ({icao}) [Meteostat {meteostat_id}]")
        frames = []

        for year in range(START_YEAR, END_YEAR + 1):
            try:
                yearly = download_station_year(meteostat_id, year)
                if yearly.empty:
                    print(f"  {year}: no file")
                    continue

                yearly = normalize_hourly_df(yearly, city, icao, meteostat_id)
                if yearly.empty:
                    print(f"  {year}: empty after parsing")
                    continue

                frames.append(yearly)
                print(f"  {year}: {len(yearly):,} rows")
            except Exception as e:
                msg = f"{city} ({icao}) year {year}: {e}"
                print(f"  {year}: FAILED -> {e}")
                failures.append(msg)

        if not frames:
            msg = f"{city} ({icao}): no data downloaded"
            print(f"FAILED {msg}")
            failures.append(msg)
            continue

        df = pd.concat(frames, ignore_index=True)
        df = df[df["timestamp"] <= CUT_OFF_INCLUSIVE].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        output_file = f"{city}_{icao}_hourly_weather_2005_2026-03-01.csv"
        output_path = os.path.join(RUN_FOLDER, output_file)
        df.to_csv(output_path, index=False)

        print(f"Saved {len(df):,} rows -> {output_file}")

    print("\nDownload complete.")

    if failures:
        print("\nFailures:")
        for item in failures:
            print(f" - {item}")


if __name__ == "__main__":
    main()
