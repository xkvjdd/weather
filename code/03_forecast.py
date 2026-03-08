# Requirements: requests, pandas, pyarrow, selenium, webdriver-manager

import math
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


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
    "LON": {"icao": "EGLC", "lat": 51.5053, "lon": 0.0553, "tz": "Europe/London"},
    "WLG": {"icao": "NZWN", "lat": -41.3272, "lon": 174.8050, "tz": "Pacific/Auckland"},
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "forecast_latest.parquet")
OBSERVATIONS_PATH = os.path.join(DATA_DIR, "observations.parquet")
OBSERVATIONS_COPY = os.path.join(DATA_DIR, "observations_copy.parquet")

HTTP_TIMEOUT = 15
PAGE_TIMEOUT = 12
MAX_WORKERS = 3

CHROMEDRIVER_PATH = ChromeDriverManager().install()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    )
}

FORECAST_COLS = [
    "airport", "icao", "forecast_date_local", "pulled_at_local",
    "forecast_source_1", "forecast_source_2", "forecast_source_3",
    "forecast_avg_max", "observed_max_today",
    "source_1_name", "source_2_name", "source_3_name",
]


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
    xs = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not xs:
        return None
    return round(sum(xs) / len(xs), 3)


def extract_number(text: str) -> float | None:
    if text is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def snapshot_observations() -> None:
    if os.path.exists(OBSERVATIONS_PATH):
        shutil.copy2(OBSERVATIONS_PATH, OBSERVATIONS_COPY)
        print(f"Snapshotted observations -> {OBSERVATIONS_COPY}")
    else:
        print(f"WARNING: observations.parquet not found at {OBSERVATIONS_PATH}, copy skipped.")


def cleanup_observations_copy() -> None:
    if os.path.exists(OBSERVATIONS_COPY):
        os.remove(OBSERVATIONS_COPY)
        print(f"Deleted observations copy: {OBSERVATIONS_COPY}")


def load_observed_max_today(airport: str, local_date: str) -> float | None:
    if not os.path.exists(OBSERVATIONS_COPY):
        return None
    try:
        obs = pd.read_parquet(OBSERVATIONS_COPY)
        mask = (
            (obs["airport"] == airport) &
            (obs["timestamp_local"].astype(str).str[:10] == local_date)
        )
        todays = obs.loc[mask, "temp"].dropna()
        if todays.empty:
            return None
        return round(float(todays.max()), 3)
    except Exception as e:
        print(f"    [{airport}] WARNING: could not read observations_copy: {e}")
        return None


def make_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.page_load_strategy = "eager"
    return webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)


def dismiss_common_popups(driver: webdriver.Chrome) -> None:
    selectors = [
        "#bbccookies-continue-button",
        "button[aria-label='Close']",
    ]
    for sel in selectors:
        try:
            for elem in driver.find_elements(By.CSS_SELECTOR, sel)[:2]:
                driver.execute_script("arguments[0].click();", elem)
        except Exception:
            pass


# UPDATED S1 (Wunderground forecast via requests)

def fetch_wunderground_forecast_high_selenium(
    driver: webdriver.Chrome,
    icao: str,
    local_date: str,
) -> float | None:
    try:
        url = f"https://www.wunderground.com/weather/{icao}"
        resp = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        html = resp.text

        m = re.search(r'"temperatureMax":\[(\-?\d+)', html)
        if not m:
            print(f"    [{icao}] WARNING: temperatureMax not found")
            return None

        temp_f = float(m.group(1))
        temp_c = (temp_f - 32.0) * 5.0 / 9.0

        return round(temp_c, 3)

    except Exception as e:
        print(f"    [{icao}] WARNING: Wunderground parse failed: {e}")
        return None


def fetch_bbc_today_high_requests(
    airport: str,
    session: requests.Session,
) -> float | None:
    url = BBC_WEATHER_URLS.get(airport)
    if not url:
        return None

    try:
        resp = session.get(url, timeout=HTTP_TIMEOUT)
        html = resp.text
    except Exception:
        return None

    m = re.search(r"High\s+(-?\d+)", html)
    if m:
        return float(m.group(1))

    return None


def process_airport(airport: str, meta: dict) -> dict | None:
    tz_name = meta["tz"]
    icao = meta["icao"]
    local_date = today_local_str(tz_name)
    pulled_at_local = now_local_naive(tz_name)

    driver = make_driver()
    session = requests.Session()
    session.headers.update(HEADERS)

    try:

        observed_max = load_observed_max_today(airport, local_date)

        src1 = fetch_wunderground_forecast_high_selenium(driver, icao, local_date)
        src2 = fetch_bbc_today_high_requests(airport, session)

        src3 = math.nan
        avg_val = mean_ignore_none([src1, src2, src3])

        print(f"[{airport}] S1={src1} S2={src2} AVG={avg_val}")

        return {
            "airport": airport,
            "icao": icao,
            "forecast_date_local": local_date,
            "pulled_at_local": pulled_at_local,
            "forecast_source_1": src1,
            "forecast_source_2": src2,
            "forecast_source_3": src3,
            "forecast_avg_max": avg_val,
            "observed_max_today": observed_max,
            "source_1_name": "wunderground_forecast_high",
            "source_2_name": "bbc_today_high",
            "source_3_name": "nan",
        }

    finally:
        session.close()
        driver.quit()


def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=FORECAST_COLS)
    return pd.read_parquet(path)


def main() -> None:

    snapshot_observations()

    rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_airport, airport, meta): airport
            for airport, meta in AIRPORTS.items()
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)

    cleanup_observations_copy()

    if not rows:
        return

    new_df = pd.DataFrame(rows)

    existing = load_existing(OUTPUT_PATH)
    combined = pd.concat([existing, new_df], ignore_index=True)

    combined.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(combined)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
