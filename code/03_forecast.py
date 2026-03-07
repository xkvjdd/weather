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
    "ATL": "https://www.bbc.com/weather/4199556",  # Hartsfield–Jackson Atlanta International Airport
    "NYC": "https://www.bbc.com/weather/5123698",  # La Guardia Airport
    "CHI": "https://www.bbc.com/weather/4887479",  # Chicago O'Hare International Airport
    "DAL": "https://www.bbc.com/weather/4684888",  # Dallas (TX) page; observation station is Dallas Love Field
    "SEA": "https://www.bbc.com/weather/5809876",  # Seattle-Tacoma International Airport
    "MIA": "https://www.bbc.com/weather/4164181",  # Miami International Airport
    "TOR": "https://www.bbc.com/weather/6296338",  # Toronto Pearson International Airport
    "PAR": "https://www.bbc.com/weather/6269554",  # Paris Charles de Gaulle Airport
    "SEL": "https://www.bbc.com/weather/1835848",  # Seoul fallback; BBC did not expose a working Incheon airport page
    "ANK": "https://www.bbc.com/weather/6299725",  # Ankara Esenboğa International Airport
    "BUE": "https://www.bbc.com/weather/6300524",  # Ministro Pistarini International Airport (Ezeiza)
    "LON": "https://www.bbc.com/weather/6296599",  # London City Airport
    "WLG": "https://www.bbc.com/weather/6244688",  # Wellington International Airport
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Observations snapshot
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Selenium
# ---------------------------------------------------------------------------

def make_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-sync")
    options.add_argument("--disable-default-apps")
    options.add_argument("--no-first-run")
    options.add_argument("--mute-audio")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"--user-agent={HEADERS['User-Agent']}")
    options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    })
    options.page_load_strategy = "eager"
    return webdriver.Chrome(
        service=Service(CHROMEDRIVER_PATH),
        options=options,
    )


def dismiss_common_popups(driver: webdriver.Chrome) -> None:
    selectors = [
        "#bbccookies-continue-button",
        "button[aria-label='Close']",
        "button[aria-label='close']",
        "button[title='Close']",
        "button[title='close']",
        ".fc-button.fc-cta-consent.fc-primary-button",
    ]
    for sel in selectors:
        try:
            for elem in driver.find_elements(By.CSS_SELECTOR, sel)[:2]:
                try:
                    driver.execute_script("arguments[0].click();", elem)
                    time.sleep(0.1)
                except Exception:
                    pass
        except Exception:
            pass


# ---------------------------------------------------------------------------
# S1 — Wunderground
# ---------------------------------------------------------------------------

def fetch_wunderground_forecast_high_selenium(
    driver: webdriver.Chrome,
    icao: str,
    local_date: str,
) -> float | None:
    url = f"https://www.wunderground.com/history/daily/{icao}/date/{local_date}"
    driver.get(url)

    try:
        WebDriverWait(driver, PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(0.8)
        dismiss_common_popups(driver)
    except Exception:
        print(f"    [{icao}] WARNING: body did not load")
        return None

    unit = "F"
    try:
        body_text = driver.find_element(By.TAG_NAME, "body").text
        if "°C" in body_text:
            unit = "C"
    except Exception:
        pass

    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "div.row.collapse")
        for row in rows:
            try:
                cols = row.find_elements(By.CSS_SELECTOR, "div.columns")
                if len(cols) < 3:
                    continue
                label = cols[0].text.strip().lower()
                if label != "high":
                    continue
                span = cols[1].find_element(By.CSS_SELECTOR, "span.wu-value, span[class*='wu-value']")
                val = clean_temp_to_celsius(extract_number(span.text), unit)
                if val is not None:
                    return val
            except Exception:
                continue
    except Exception:
        pass

    try:
        html = driver.page_source
        patterns = [
            r'High.*?wu-value[^>]*>(-?\d+(?:\.\d+)?)</span>',
            r'High.*?wu-value-to[^>]*>(-?\d+(?:\.\d+)?)</span>',
        ]
        for pat in patterns:
            m = re.search(pat, html, flags=re.IGNORECASE | re.DOTALL)
            if m:
                val = clean_temp_to_celsius(m.group(1), unit)
                if val is not None:
                    return val
    except Exception:
        pass

    print(f"    [{icao}] WARNING: could not parse Wunderground forecast high for {local_date}")
    return None


# ---------------------------------------------------------------------------
# S2 — BBC via requests
# ---------------------------------------------------------------------------

def fetch_bbc_today_high_requests(
    airport: str,
    session: requests.Session,
) -> float | None:
    url = BBC_WEATHER_URLS.get(airport)
    if not url:
        print(f"    [{airport}] WARNING: no BBC URL configured")
        return None

    try:
        resp = session.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"    [{airport}] WARNING: BBC request failed: {e}")
        return None

    patterns = [
        r"Today\s*,.*?High\s+(-?\d+(?:\.\d+)?)°",
        r"Tonight\s*,.*?High\s+(-?\d+(?:\.\d+)?)°",
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return clean_temp_to_celsius(m.group(1), "C")

    try:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                return clean_temp_to_celsius(m.group(1), "C")
    except Exception:
        pass

    print(f"    [{airport}] WARNING: could not parse BBC today high")
    return None


# ---------------------------------------------------------------------------
# Purge old rows
# ---------------------------------------------------------------------------

def purge_old_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep_mask = df.apply(
        lambda row: row["forecast_date_local"] == today_local_str(
            AIRPORTS.get(row["airport"], {}).get("tz", "UTC")
        ),
        axis=1,
    )
    dropped = int((~keep_mask).sum())
    if dropped:
        print(f"  Purged {dropped} row(s) from previous local days")
    return df[keep_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-airport worker
# ---------------------------------------------------------------------------

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
        print(f"  [{airport}] Observed max today = {observed_max}")

        src1 = None
        try:
            src1 = fetch_wunderground_forecast_high_selenium(driver, icao, local_date)
        except Exception as e:
            print(f"  [{airport}] S1 Wunderground FAILED: {e}")

        src2 = None
        try:
            src2 = fetch_bbc_today_high_requests(airport, session)
        except Exception as e:
            print(f"  [{airport}] S2 BBC FAILED: {e}")

        src3 = math.nan
        avg_val = mean_ignore_none([src1, src2, src3])

        print(f"  [{airport}] S1={src1}  S2={src2}  S3={src3}  AVG={avg_val}")

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

    except Exception as e:
        print(f"  [{airport}] FAILED: {e}")
        return None

    finally:
        session.close()
        driver.quit()


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def load_existing(path: str) -> pd.DataFrame:
    cols = [
        "airport", "icao", "forecast_date_local", "pulled_at_local",
        "forecast_source_1", "forecast_source_2", "forecast_source_3",
        "forecast_avg_max", "observed_max_today",
        "source_1_name", "source_2_name", "source_3_name",
    ]
    if os.path.exists(path):
        df = pd.read_parquet(path)
        keep = [c for c in cols if c in df.columns]
        if keep:
            return df[keep].copy()
    return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    snapshot_observations()

    rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_airport, airport, meta): airport
            for airport, meta in AIRPORTS.items()
        }
        for future in as_completed(futures):
            airport = futures[future]
            try:
                result = future.result()
                if result is not None:
                    rows.append(result)
            except Exception as e:
                print(f"  [{airport}] Unhandled exception: {e}")

    cleanup_observations_copy()

    if not rows:
        print("No rows fetched; nothing saved.")
        return

    new_df = pd.DataFrame(rows)
    existing = load_existing(OUTPUT_PATH)
    existing = purge_old_forecasts(existing)
    combined = pd.concat([existing, new_df], ignore_index=True)

    combined["pulled_at_local"] = pd.to_datetime(combined["pulled_at_local"], errors="coerce")
    for col in [
        "forecast_source_1", "forecast_source_2", "forecast_source_3",
        "forecast_avg_max", "observed_max_today"
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
    print(f"\nSaved {len(combined):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
