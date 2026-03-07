# Requirements: requests, pandas, pyarrow, selenium, beautifulsoup4

import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

AIRPORTS = {
    "ATL": {"icao": "KATL", "lat": 33.6407, "lon": -84.4277, "tz": "America/New_York"},
    "NYC": {"icao": "KLGA", "lat": 40.7769, "lon": -73.8740, "tz": "America/New_York"},
    "CHI": {"icao": "KORD", "lat": 41.9742, "lon": -87.9073, "tz": "America/Chicago"},
    "DAL": {"icao": "KDAL", "lat": 32.8471, "lon": -96.8517, "tz": "America/Chicago"},
    "SEA": {"icao": "KSEA", "lat": 47.4502, "lon": -122.3088, "tz": "America/Los_Angeles"},
    "MIA": {"icao": "KMIA", "lat": 25.7959, "lon": -80.2870, "tz": "America/New_York"},
    "TOR": {"icao": "CYYZ", "lat": 43.6777, "lon": -79.6248, "tz": "America/Toronto"},
    "PAR": {"icao": "LFPG", "lat": 49.0097, "lon": 2.5479,   "tz": "Europe/Paris"},
    "SEL": {"icao": "RKSI", "lat": 37.4602, "lon": 126.4407, "tz": "Asia/Seoul"},
    "ANK": {"icao": "LTAC", "lat": 40.1281, "lon": 32.9951,  "tz": "Europe/Istanbul"},
    "BUE": {"icao": "SAEZ", "lat": -34.8222,"lon": -58.5358, "tz": "America/Argentina/Buenos_Aires"},
    "LON": {"icao": "EGLL", "lat": 51.4700, "lon": -0.4543,  "tz": "Europe/London"},
    "WLG": {"icao": "NZWN", "lat": -41.3272,"lon": 174.8050, "tz": "Pacific/Auckland"},
}

# S3 regional model — best available NWP model per location
S3_MODEL = {
    "ATL": "gem_seamless",
    "NYC": "gem_seamless",
    "CHI": "gem_seamless",
    "DAL": "gem_seamless",
    "SEA": "gem_seamless",
    "MIA": "gem_seamless",
    "TOR": "gem_seamless",
    "LON": "icon_seamless",
    "PAR": "icon_seamless",
    "ANK": "icon_seamless",
    "SEL": "jma_seamless",
    "BUE": "bom_access_global",
    "WLG": "bom_access_global",
}

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT        = os.path.dirname(SCRIPT_DIR)
DATA_DIR         = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH      = os.path.join(DATA_DIR, "forecast_latest.parquet")
OBSERVATIONS_PATH = os.path.join(DATA_DIR, "observations.parquet")

HTTP_TIMEOUT  = 30
SLEEP_SECONDS = 0.5
PAGE_TIMEOUT  = 20

HEADERS_API = {
    "User-Agent": "airport-weather-dashboard/1.0 (research; contact via github)"
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_api_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS_API)
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


def max_ignore_none(a: float | None, b: float | None) -> float | None:
    vals = [v for v in [a, b] if v is not None]
    return round(max(vals), 3) if vals else None


# ---------------------------------------------------------------------------
# Observed max today — read from observations.parquet
# Filters to rows for the given airport whose timestamp_local falls on
# today in that airport's local timezone, then returns the max temp seen.
# ---------------------------------------------------------------------------

def load_observed_max_today(airport: str, tz_name: str) -> float | None:
    if not os.path.exists(OBSERVATIONS_PATH):
        print(f"    WARNING: observations.parquet not found at {OBSERVATIONS_PATH}")
        return None

    try:
        obs = pd.read_parquet(OBSERVATIONS_PATH)
    except Exception as e:
        print(f"    WARNING: could not read observations.parquet: {e}")
        return None

    today = today_local_str(tz_name)

    mask = (
        (obs["airport"] == airport) &
        (obs["timestamp_local"].astype(str).str[:10] == today)
    )
    todays = obs.loc[mask, "temp"].dropna()

    if todays.empty:
        print(f"    WARNING: no observations for {airport} on {today}")
        return None

    return round(float(todays.max()), 3)


# ---------------------------------------------------------------------------
# S1 — Wunderground almanac scrape via Selenium
# URL: wunderground.com/history/daily/{ICAO}/date/{YYYY-MM-DD}
# Today's date is in the URL — can never bleed into tomorrow.
# Waits for the Angular app to render, then finds the almanac High row
# and reads the forecast column value.
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
    options.add_argument("--window-size=1400,1200")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    )
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    }
    options.add_experimental_option("prefs", prefs)
    options.page_load_strategy = "eager"
    return webdriver.Chrome(options=options)


def fetch_wunderground_almanac_high_selenium(
    driver: webdriver.Chrome,
    icao: str,
    tz_name: str,
) -> float | None:
    today = today_local_str(tz_name)
    url   = f"https://www.wunderground.com/history/daily/{icao}/date/{today}"

    driver.get(url)

    # Wait for the almanac section to render
    try:
        WebDriverWait(driver, PAGE_TIMEOUT).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "lib-city-history-almanac")
            )
        )
        # Give Angular a moment to populate the values
        time.sleep(2)
    except Exception:
        print(f"    WARNING: Wunderground almanac section did not load for {icao}")
        return None

    # Detect unit from page
    unit = "F"
    try:
        page_text = driver.find_element(By.TAG_NAME, "body").text
        if "°C" in page_text or "\u2103" in page_text:
            unit = "C"
    except Exception:
        pass

    # Strategy: find the almanac table rows, locate "High" row,
    # grab the first wu-value span in the Forecast column (first td after label)
    try:
        # Find all row collapse divs inside the almanac component
        almanac = driver.find_element(By.CSS_SELECTOR, "lib-city-history-almanac")

        # Get all text nodes that say "High" and find the associated value
        # The structure is: div.row.collapse > div.columns "High" | div.columns value
        rows = almanac.find_elements(By.CSS_SELECTOR, "div.row.collapse")

        for row in rows:
            try:
                label_divs = row.find_elements(By.CSS_SELECTOR, "div.columns")
                if not label_divs:
                    continue
                label_text = label_divs[0].text.strip()
                if label_text.lower() != "high":
                    continue

                # The forecast value is in the second column (index 1)
                if len(label_divs) < 2:
                    continue

                value_div = label_divs[1]
                # Get the wu-value span inside it
                span = value_div.find_element(
                    By.CSS_SELECTOR, "span.wu-value, span[class*='wu-value']"
                )
                val_text = span.text.strip()
                if val_text:
                    result = clean_temp_to_celsius(float(val_text), unit)
                    if result is not None:
                        return result
            except Exception:
                continue

    except Exception as e:
        print(f"    WARNING: Wunderground almanac parse error for {icao}: {e}")

    # Fallback: XPath directly targeting High row forecast value
    try:
        xpath = (
            "//lib-city-history-almanac"
            "//div[contains(@class,'row') and contains(@class,'collapse')]"
            "[.//div[normalize-space(text())='High']]"
            "//span[contains(@class,'wu-value')]"
        )
        spans = driver.find_elements(By.XPATH, xpath)
        for span in spans:
            try:
                val = float(span.text.strip())
                result = clean_temp_to_celsius(val, unit)
                if result is not None:
                    return result
            except ValueError:
                continue
    except Exception as e:
        print(f"    WARNING: Wunderground XPath fallback failed for {icao}: {e}")

    print(f"    WARNING: Wunderground — could not parse almanac high for {icao} on {today}")
    return None


# ---------------------------------------------------------------------------
# Open-Meteo daily max — shared fetch, model parameterised
# start_date=end_date=today (local) prevents any cross-day bleed.
# ---------------------------------------------------------------------------

def fetch_open_meteo_daily_max(
    session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
    model: str,
) -> float | None:
    url   = "https://api.open-meteo.com/v1/forecast"
    today = today_local_str(tz_name)
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "daily":            "temperature_2m_max",
        "start_date":       today,
        "end_date":         today,
        "timezone":         tz_name,
        "temperature_unit": "celsius",
        "models":           model,
    }
    r     = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data  = r.json()
    dates = data.get("daily", {}).get("time", [])
    vals  = data.get("daily", {}).get("temperature_2m_max", [])

    for d, v in zip(dates, vals):
        if d == today and v is not None:
            return clean_temp_to_celsius(v, "C")

    print(f"    WARNING: Open-Meteo [{model}] — no match for {today}, got: {dates}")
    return None


# ---------------------------------------------------------------------------
# Purge rows from previous local days
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
# Per-airport orchestration
# ---------------------------------------------------------------------------

def fetch_airport_forecasts(
    driver: webdriver.Chrome,
    api_session: requests.Session,
    airport: str,
    meta: dict,
) -> dict:
    lat, lon, tz_name, icao = meta["lat"], meta["lon"], meta["tz"], meta["icao"]
    s3_model = S3_MODEL[airport]

    pulled_at_local     = now_local_naive(tz_name)
    forecast_date_local = today_local_str(tz_name)

    # Observed max so far today from observations.parquet
    observed_max = load_observed_max_today(airport, tz_name)
    print(f"  Observed max today = {observed_max}")

    # S1 — Wunderground almanac forecast high (Selenium)
    src1 = None
    try:
        src1 = fetch_wunderground_almanac_high_selenium(driver, icao, tz_name)
    except Exception as e:
        print(f"  S1 Wunderground FAILED: {e}")

    # S2 — max(ECMWF forecast, observed max today)
    src2_forecast = None
    try:
        src2_forecast = fetch_open_meteo_daily_max(
            api_session, lat, lon, tz_name, "ecmwf_ifs025"
        )
    except Exception as e:
        print(f"  S2 ECMWF FAILED: {e}")
    src2 = max_ignore_none(src2_forecast, observed_max)

    # S3 — max(regional model forecast, observed max today)
    src3_forecast = None
    try:
        src3_forecast = fetch_open_meteo_daily_max(
            api_session, lat, lon, tz_name, s3_model
        )
    except Exception as e:
        print(f"  S3 {s3_model} FAILED: {e}")
    src3 = max_ignore_none(src3_forecast, observed_max)

    return {
        "airport":             airport,
        "icao":                icao,
        "forecast_date_local": forecast_date_local,
        "pulled_at_local":     pulled_at_local,
        "forecast_source_1":   src1,
        "forecast_source_2":   src2,
        "forecast_source_3":   src3,
        "forecast_avg_max":    mean_ignore_none([src1, src2, src3]),
        "source_1_name":       "wunderground_almanac_forecast_high",
        "source_2_name":       f"max(ecmwf_ifs025,observed_today)",
        "source_3_name":       f"max({s3_model},observed_today)",
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
        df   = pd.read_parquet(path)
        keep = [c for c in cols if c in df.columns]
        if keep:
            return df[keep].copy()
    return pd.DataFrame(columns=cols)


def main() -> None:
    api_session = make_api_session()
    driver      = make_driver()
    rows: list[dict] = []

    try:
        for airport, meta in AIRPORTS.items():
            s3_model = S3_MODEL[airport]
            print(
                f"\nFetching {airport} ({meta['icao']}) "
                f"[S1=wunderground | S2=ecmwf+obs | S3={s3_model}+obs] "
                f"local date: {today_local_str(meta['tz'])}"
            )
            try:
                row = fetch_airport_forecasts(driver, api_session, airport, meta)
                rows.append(row)
                print(
                    f"  S1 Wunderground      = {row['forecast_source_1']}\n"
                    f"  S2 ECMWF+obs         = {row['forecast_source_2']}\n"
                    f"  S3 {s3_model}+obs = {row['forecast_source_3']}\n"
                    f"  AVG                  = {row['forecast_avg_max']}"
                )
            except Exception as e:
                print(f"  FAILED: {e}")

            time.sleep(SLEEP_SECONDS)

    finally:
        driver.quit()

    if not rows:
        print("No rows fetched; nothing saved.")
        return

    new_df   = pd.DataFrame(rows)
    existing = load_existing(OUTPUT_PATH)
    existing = purge_old_forecasts(existing)
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
    print(f"\nSaved {len(combined):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
