# Requirements:
# requests, pandas, pyarrow, herbie-data, xarray, cfgrib, eccodes
#
# If needed:
# pip uninstall herbie
# pip install herbie-data xarray cfgrib eccodes

import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import xarray as xr
from herbie import Herbie


AIRPORTS = {
    "ATL": {"icao": "KATL", "tz": "America/New_York"},
    "NYC": {"icao": "KLGA", "tz": "America/New_York"},
    "CHI": {"icao": "KORD", "tz": "America/Chicago"},
    "DAL": {"icao": "KDAL", "tz": "America/Chicago"},
    "SEA": {"icao": "KSEA", "tz": "America/Los_Angeles"},
    "MIA": {"icao": "KMIA", "tz": "America/New_York"},
    "TOR": {"icao": "CYYZ", "tz": "America/Toronto"},
    "PAR": {"icao": "LFPG", "tz": "Europe/Paris"},
    "SEL": {"icao": "RKSI", "tz": "Asia/Seoul"},
    "ANK": {"icao": "LTAC", "tz": "Europe/Istanbul"},
    "BUE": {"icao": "SAEZ", "tz": "America/Argentina/Buenos_Aires"},
    "LON": {"icao": "EGLC", "tz": "Europe/London"},
    "WLG": {"icao": "NZWN", "tz": "Pacific/Auckland"},
}

# METAR-TAF airport coordinates for US airports only
HRRR_COORDS = {
    "ATL": {"lat": 33.63670, "lon": -84.42790},   # KATL
    "NYC": {"lat": 40.77720, "lon": -73.87260},   # KLGA
    "CHI": {"lat": 41.97690, "lon": -87.90810},   # KORD
    "DAL": {"lat": 32.84590, "lon": -96.85090},   # KDAL
    "SEA": {"lat": 47.44990, "lon": -122.31180},  # KSEA
    "MIA": {"lat": 25.79540, "lon": -80.29010},   # KMIA
}


BBC_WEATHER_URLS = {
    "ATL": "https://www.bbc.com/weather/4199556",
    "NYC": "https://www.bbc.com/weather/5123698",
    "CHI": "https://www.bbc.com/weather/4887479",
    "DAL": "https://www.bbc.com/weather/4684888",
    "SEA": "https://www.bbc.com/weather/5809876",
    "MIA": "https://www.bbc.com/weather/4164181",
    "TOR": "https://www.bbc.com/weather/6296338",
    "PAR": "https://www.bbc.com/weather/6269554",
    "SEL": "https://www.bbc.com/weather/1835848",
    "ANK": "https://www.bbc.com/weather/6299725",
    "BUE": "https://www.bbc.com/weather/6300524",
    "LON": "https://www.bbc.com/weather/6296599",
    "WLG": "https://www.bbc.com/weather/6244688",
}


ACCUWEATHER_URLS = {
    "ATL": "https://www.accuweather.com/en/us/atlanta/30303/weather-forecast/348181",
    "NYC": "https://www.accuweather.com/en/us/new-york/10021/weather-forecast/349727",
    "CHI": "https://www.accuweather.com/en/us/chicago/60608/weather-forecast/348308",
    "DAL": "https://www.accuweather.com/en/us/dallas/75202/weather-forecast/351194",
    "SEA": "https://www.accuweather.com/en/us/seattle/98104/weather-forecast/351409",
    "MIA": "https://www.accuweather.com/en/us/miami/33128/weather-forecast/347936",
    "TOR": "https://www.accuweather.com/en/ca/toronto/m5h/weather-forecast/55488",
    "PAR": "https://www.accuweather.com/en/fr/paris/623/weather-forecast/623",
    "SEL": "https://www.accuweather.com/en/kr/seoul/226081/weather-forecast/226081",
    "ANK": "https://www.accuweather.com/en/tr/ankara/318251/weather-forecast/318251",
    "BUE": "https://www.accuweather.com/en/ar/buenos-aires/7894/weather-forecast/7894",
    "LON": "https://www.accuweather.com/en/gb/london/ec4a-2/weather-forecast/328328",
    "WLG": "https://www.accuweather.com/en/nz/wellington/250938/weather-forecast/250938",
}


HEADERS = {"User-Agent": "Mozilla/5.0"}

DATA_DIR = "data/dashboard"
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "forecast_latest.parquet")

HTTP_TIMEOUT = 15

# Keep this low on Windows because cfgrib/eccodes can be unstable with parallel GRIB reads.
MAX_WORKERS = 1


def now_local_naive(tz):
    return pd.Timestamp(datetime.now(ZoneInfo(tz)).replace(tzinfo=None))


def today_local_str(tz):
    return datetime.now(ZoneInfo(tz)).date().isoformat()


def f_to_c(temp_f):
    return round((float(temp_f) - 32) * 5 / 9, 3)


def clean_temp_c(x):
    if x is None or pd.isna(x):
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if x < -60 or x > 60:
        return None
    return round(x, 3)


def mean_ignore_none(values):
    xs = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not xs:
        return None
    return round(sum(xs) / len(xs), 3)


# ------------------------------------------------
# S1 HRRR (C) - US airports only
# ------------------------------------------------

def fetch_hrrr_today_high_c(airport, tz):
    coords = HRRR_COORDS.get(airport)
    if not coords:
        return None

    lat = coords["lat"]
    lon = coords["lon"]

    now_utc = datetime.now(ZoneInfo("UTC"))
    today_local = now_utc.astimezone(ZoneInfo(tz)).date()

    warnings.filterwarnings("ignore")

    # Try latest likely available run first, then fall back
    for lag in [1, 2, 3, 4]:
        run = (now_utc - timedelta(hours=lag)).replace(
            tzinfo=None, minute=0, second=0, microsecond=0
        )

        vals = []

        try:
            for fxx in range(19):
                H = Herbie(
                    run,
                    model="hrrr",
                    product="sfc",
                    fxx=fxx,
                    priority="aws",
                    verbose=False,
                )

                grib_file = H.download(":TMP:2 m above ground:")

                ds = xr.open_dataset(
                    grib_file,
                    engine="cfgrib",
                    backend_kwargs={"indexpath": ""},
                )

                try:
                    # HRRR longitude may be 0..360 depending on how cfgrib opens it
                    ds_lon_max = float(np.nanmax(ds.longitude.values))
                    target_lon = lon % 360 if ds_lon_max > 180 else lon

                    dist = ((ds.latitude - lat) ** 2 + (ds.longitude - target_lon) ** 2).values
                    k = int(np.argmin(dist))
                    nx = ds.latitude.shape[1]
                    y, x = divmod(k, nx)

                    temp_c = float(ds["t2m"].isel(y=y, x=x).item() - 273.15)

                    valid_date_local = (
                        (run + timedelta(hours=fxx))
                        .replace(tzinfo=ZoneInfo("UTC"))
                        .astimezone(ZoneInfo(tz))
                        .date()
                    )

                    if valid_date_local == today_local:
                        vals.append(temp_c)
                finally:
                    ds.close()

            if vals:
                return clean_temp_c(max(vals))

        except Exception as e:
            print(f"[HRRR FAIL {airport} lag={lag}] {e}")
            continue

    return None


# ------------------------------------------------
# S2 BBC (already C)
# ------------------------------------------------

def fetch_bbc_today_high(airport):
    url = BBC_WEATHER_URLS.get(airport)
    if not url:
        return None

    try:
        html = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT).text

        i = html.find("wr-day-temperature__high-value")
        if i == -1:
            return None

        snippet = html[i:i + 120]
        m = re.search(r"([0-9]+)", snippet)
        if m:
            return float(m.group(1))

    except Exception as e:
        print(f"[BBC FAIL {airport}] {e}")

    return None


# ------------------------------------------------
# S3 ACCUWEATHER (F -> C)
# ------------------------------------------------

def fetch_accuweather_today_high(airport):
    url = ACCUWEATHER_URLS.get(airport)
    if not url:
        return None

    try:
        html = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT).text

        i = html.find("temp-hi")
        if i == -1:
            return None

        snippet = html[i:i + 80]
        m = re.search(r"([0-9]{2,3})", snippet)
        if not m:
            return None

        temp_f = float(m.group(1))
        return f_to_c(temp_f)

    except Exception as e:
        print(f"[ACCUWEATHER FAIL {airport}] {e}")

    return None


def process_airport(airport, meta):
    tz = meta["tz"]
    icao = meta["icao"]

    pulled_at = now_local_naive(tz)

    s1 = clean_temp_c(fetch_hrrr_today_high_c(airport, tz))
    s2 = clean_temp_c(fetch_bbc_today_high(airport))
    s3 = clean_temp_c(fetch_accuweather_today_high(airport))

    avg = mean_ignore_none([s1, s2, s3])

    print(airport, "S1=", s1, "S2=", s2, "S3=", s3, "AVG=", avg)

    return {
        "airport": airport,
        "icao": icao,
        "forecast_date_local": today_local_str(tz),
        "pulled_at_local": pulled_at,
        "forecast_source_1": s1,
        "forecast_source_2": s2,
        "forecast_source_3": s3,
        "forecast_avg_max": avg,
        "source_1_name": "hrrr",
        "source_2_name": "bbc",
        "source_3_name": "accuweather",
    }


def main():
    rows = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_airport, airport, meta): airport
            for airport, meta in AIRPORTS.items()
        }

        for future in as_completed(futures):
            r = future.result()
            if r:
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("Saved", len(df), "rows")


if __name__ == "__main__":
    main()
