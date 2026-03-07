# Requirements: requests, pandas, pyarrow, selenium, webdriver-manager, beautifulsoup4

import os
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
    "PAR": {"icao": "LFPG", "lat": 49.0097, "lon": 2.5479,   "tz": "Europe/Paris"},
    "SEL": {"icao": "RKSI", "lat": 37.4602, "lon": 126.4407, "tz": "Asia/Seoul"},
    "ANK": {"icao": "LTAC", "lat": 40.1281, "lon": 32.9951,  "tz": "Europe/Istanbul"},
    "BUE": {"icao": "SAEZ", "lat": -34.8222,"lon": -58.5358, "tz": "America/Argentina/Buenos_Aires"},
    "LON": {"icao": "EGLL", "lat": 51.4700, "lon": -0.4543,  "tz": "Europe/London"},
    "WLG": {"icao": "NZWN", "lat": -41.3272,"lon": 174.8050, "tz": "Pacific/Auckland"},
}

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

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT         = os.path.dirname(SCRIPT_DIR)
DATA_DIR          = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH        = os.path.join(DATA_DIR, "forecast_latest.parquet")
OBSERVATIONS_PATH  = os.path.join(DATA_DIR, "observations.parquet")
OBSERVATIONS_COPY  = os.path.join(DATA_DIR, "observations_copy.parquet")

HTTP_TIMEOUT  = 20
PAGE_TIMEOUT  = 15

# Install ChromeDriver once before threads start — avoids concurrent installs
CHROMEDRIVER_PATH = ChromeDriverManager().install()

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
# Observations snapshot — copy once at startup, read copy throughout
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


def load_observed_max_today(airport: str, tz_name: str) -> float | None:
    if not os.path.exists(OBSERVATIONS_COPY):
        return None
    try:
        obs   = pd.read_parquet(OBSERVATIONS_COPY)
        today = today_local_str(tz_name)
        mask  = (
            (obs["airport"] == airport) &
            (obs["timestamp_local"].astype(str).str[:10] == today)
        )
        todays = obs.loc[mask, "temp"].dropna()
        if todays.empty:
            print(f"    [{airport}] WARNING: no observations for {today}")
            return None
        return round(float(todays.max()), 3)
    except Exception as e:
        print(f"    [{airport}] WARNING: could not read observations_copy: {e}")
        return None


# ---------------------------------------------------------------------------
# Selenium — one driver per thread, using pre-installed ChromeDriver
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
    options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    })
    options.page_load_strategy = "eager"
    return webdriver.Chrome(
        service=Service(CHROMEDRIVER_PATH),
        options=options,
    )


def fetch_wunderground_almanac_high_selenium(
    driver: webdriver.Chrome,
    icao: str,
    tz_name: str,
) -> float | None:
    today = today_local_str(tz_name)
    url   = f"https://www.wunderground.com/history/daily/{icao}/date/{today}"
    driver.get(url)

    try:
        WebDriverWait(driver, PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "lib-city-history-almanac"))
        )
        time.sleep(1.5)
    except Exception:
        print(f"    [{icao}] WARNING: almanac section did not load")
        return None

    unit = "F"
    try:
        if "°C" in driver.find_element(By.TAG_NAME, "body").text:
            unit = "C"
    except Exception:
        pass

    # Primary parse
    try:
        almanac = driver.find_element(By.CSS_SELECTOR, "lib-city-history-almanac")
        for row in almanac.find_elements(By.CSS_SELECTOR, "div.row.collapse"):
            try:
                cols = row.find_elements(By.CSS_SELECTOR, "div.columns")
                if not cols or cols[0].text.strip().lower() != "high" or len(cols) < 2:
                    continue
                span = cols[1].find_element(By.CSS_SELECTOR, "span.wu-value, span[class*='wu-value']")
                val  = clean_temp_to_celsius(float(span.text.strip()), unit)
                if val is not None:
                    return val
            except Exception:
                continue
    except Exception as e:
        print(f"    [{icao}] WARNING: almanac parse error: {e}")

    # XPath fallback
    try:
        xpath = (
            "//lib-city-history-almanac"
            "//div[contains(@class,'row') and contains(@class,'collapse')]"
            "[.//div[normalize-space(text())='High']]"
            "//span[contains(@class,'wu-value')]"
        )
        for span in driver.find_elements(By.XPATH, xpath):
            val = clean_temp_to_celsius(float(span.text.strip()), unit)
            if val is not None:
                return val
    except Exception:
        pass

    print(f"    [{icao}] WARNING: could not parse almanac high for {today}")
    return None


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------

def fetch_open_meteo_daily_max(
    session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
    model: str,
) -> float | None:
    today  = today_local_str(tz_name)
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
    r    = session.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    for d, v in zip(data.get("daily", {}).get("time", []), data.get("daily", {}).get("temperature_2m_max", [])):
        if d == today and v is not None:
            return clean_temp_to_celsius(v, "C")
    print(f"    WARNING: Open-Meteo [{model}] — no match for {today}")
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
# Per-airport worker — each thread owns its own driver and session
# ---------------------------------------------------------------------------

def process_airport(airport: str, meta: dict) -> dict | None:
    lat, lon, tz_name, icao = meta["lat"], meta["lon"], meta["tz"], meta["icao"]
    s3_model = S3_MODEL[airport]

    driver      = make_driver()
    api_session = make_api_session()

    try:
        pulled_at_local     = now_local_naive(tz_name)
        forecast_date_local = today_local_str(tz_name)

        observed_max = load_observed_max_today(airport, tz_name)
        print(f"  [{airport}] Observed max today = {observed_max}")

        # S1 — Wunderground
        src1 = None
        try:
            src1 = fetch_wunderground_almanac_high_selenium(driver, icao, tz_name)
        except Exception as e:
            print(f"  [{airport}] S1 Wunderground FAILED: {e}")

        # S2 — ECMWF + observed max
        src2_forecast = None
        try:
            src2_forecast = fetch_open_meteo_daily_max(api_session, lat, lon, tz_name, "ecmwf_ifs025")
        except Exception as e:
            print(f"  [{airport}] S2 ECMWF FAILED: {e}")
        src2 = max_ignore_none(src2_forecast, observed_max)

        # S3 — regional model + observed max
        src3_forecast = None
        try:
            src3_forecast = fetch_open_meteo_daily_max(api_session, lat, lon, tz_name, s3_model)
        except Exception as e:
            print(f"  [{airport}] S3 {s3_model} FAILED: {e}")
        src3 = max_ignore_none(src3_forecast, observed_max)

        print(
            f"  [{airport}] S1={src1}  S2={src2}  S3={src3}  AVG={mean_ignore_none([src1, src2, src3])}"
        )

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
            "source_2_name":       "max(ecmwf_ifs025,observed_today)",
            "source_3_name":       f"max({s3_model},observed_today)",
        }

    except Exception as e:
        print(f"  [{airport}] FAILED: {e}")
        return None

    finally:
        driver.quit()


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    snapshot_observations()

    rows: list[dict] = []

    # Run all airports in parallel — each gets its own driver + session
    # Max 13 workers (one per airport); GitHub Actions runners have enough RAM
    with ThreadPoolExecutor(max_workers=len(AIRPORTS)) as executor:
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
