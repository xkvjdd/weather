# Requirements: requests, pandas, pyarrow

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests


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
MAX_WORKERS = 6


def now_local_naive(tz):
    return pd.Timestamp(datetime.now(ZoneInfo(tz)).replace(tzinfo=None))


def today_local_str(tz):
    return datetime.now(ZoneInfo(tz)).date().isoformat()


def mean_ignore_none(values):
    xs = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not xs:
        return None
    return round(sum(xs) / len(xs), 3)


# ------------------------------------------------
# S1 WUNDERGROUND (F → C)
# ------------------------------------------------

def fetch_wunderground_forecast_high(icao):

    try:

        url = f"https://www.wunderground.com/weather/{icao}"

        html = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT).text

        m = re.search(r'"temperatureMax":\[(\-?\d+)', html)

        if not m:
            return None

        temp_f = float(m.group(1))

        temp_c = (temp_f - 32) * 5 / 9

        return round(temp_c, 3)

    except:
        return None


# ------------------------------------------------
# S2 BBC (already °C)
# ------------------------------------------------

def fetch_bbc_today_high(airport):

    url = BBC_WEATHER_URLS.get(airport)

    if not url:
        return None

    try:

        html = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT).text

        m = re.search(r'High[^0-9-]*(-?\d+)', html)

        if m:
            return float(m.group(1))

    except:
        pass

    return None


# ------------------------------------------------
# S3 ACCUWEATHER (F → C if needed)
# ------------------------------------------------

def fetch_accuweather_today_high(airport):

    url = ACCUWEATHER_URLS.get(airport)

    if not url:
        return None

    try:

        html = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT).text

        m = re.search(r'temp-hi">(\d+)', html)

        if not m:
            return None

        temp = float(m.group(1))

        # detect if page uses Fahrenheit
        if "°F" in html or "degF" in html or "/us/" in url:

            temp = (temp - 32) * 5 / 9

        return round(temp, 3)

    except:
        return None


def process_airport(airport, meta):

    tz = meta["tz"]
    icao = meta["icao"]

    pulled_at = now_local_naive(tz)

    s1 = fetch_wunderground_forecast_high(icao)
    s2 = fetch_bbc_today_high(airport)
    s3 = fetch_accuweather_today_high(airport)

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
        "source_1_name": "wunderground",
        "source_2_name": "bbc",
        "source_3_name": "accuweather",
    }


def main():

    rows = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(process_airport, a, m): a
            for a, m in AIRPORTS.items()
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
