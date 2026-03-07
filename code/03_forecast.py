# Requirements: requests, pandas, pyarrow, selenium, webdriver-manager, beautifulsoup4

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
    "PAR": {"icao": "LFPG", "lat": 49.0097, "lon": 2.5479,   "tz": "Europe/Paris"},
    "SEL": {"icao": "RKSI", "lat": 37.4602, "lon": 126.4407, "tz": "Asia/Seoul"},
    "ANK": {"icao": "LTAC", "lat": 40.1281, "lon": 32.9951,  "tz": "Europe/Istanbul"},
    "BUE": {"icao": "SAEZ", "lat": -34.8222, "lon": -58.5358, "tz": "America/Argentina/Buenos_Aires"},
    "LON": {"icao": "EGLL", "lat": 51.4700, "lon": -0.4543,  "tz": "Europe/London"},
    "WLG": {"icao": "NZWN", "lat": -41.3272, "lon": 174.8050, "tz": "Pacific/Auckland"},
}

# Meteofor location paths
# Format: country-slug / city-slug / airport-page-slug
METEOFOR_PATHS = {
    "ATL": "united-states/atlanta/airport-hartsfield-jackson",
    "NYC": "united-states/new-york/airport-la-guardia",
    "CHI": "united-states/chicago/airport-ohare",
    "DAL": "united-states/dallas/airport-love-field",
    "SEA": "united-states/seattle/airport-seattle-tacoma",
    "MIA": "united-states/miami/airport-miami",
    "TOR": "canada/toronto/airport-lester-b-pearson",
    "PAR": "france/paris/airport-charles-de-gaulle",
    "SEL": "south-korea/seoul/airport-incheon",
    "ANK": "turkey/ankara/airport-esenboga",
    "BUE": "argentina/buenos-aires/airport-ezeiza",
    "LON": "united-kingdom/london/airport-heathrow",
    "WLG": "new-zealand/wellington/airport-wellington",
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
    xs = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not xs:
        return None
    result = round(sum(xs) / len(xs), 3)
    if len(xs) > 1 and (max(xs) - min(xs)) > 3.0:
        print(f"    WARNING: Source spread > 3°C: {xs}")
    return result


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
        obs = pd.read_parquet(OBSERVATIONS_COPY)
        today = today_local_str(tz_name)
        mask = (
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
    options.add_argument("--window-size=1800,1400")
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


def dismiss_common_popups(driver: webdriver.Chrome) -> None:
    selectors = [
        "button[aria-label='Close']",
        "button[aria-label='close']",
        "button[title='Close']",
        "button[title='close']",
        ".close",
        ".modal__close",
        ".popup-close",
        ".fc-close",
        ".fc-button.fc-cta-consent.fc-primary-button",
    ]
    for sel in selectors:
        try:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            for elem in elems[:3]:
                try:
                    driver.execute_script("arguments[0].click();", elem)
                    time.sleep(0.2)
                except Exception:
                    pass
        except Exception:
            pass


# ---------------------------------------------------------------------------
# S1 — Wunderground forecast almanac high
# ---------------------------------------------------------------------------

def fetch_wunderground_forecast_high_selenium(
    driver: webdriver.Chrome,
    icao: str,
    tz_name: str,
) -> float | None:
    today = today_local_str(tz_name)
    url = f"https://www.wunderground.com/history/daily/{icao}/date/{today}"
    driver.get(url)

    try:
        WebDriverWait(driver, PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2.0)
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

    # Strategy 1: direct parse from the almanac table rows
    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "div.row.collapse")
        for row in rows:
            try:
                text = row.text.strip().lower()
                if "high" not in text:
                    continue

                cols = row.find_elements(By.CSS_SELECTOR, "div.columns")
                if len(cols) < 3:
                    continue

                label = cols[0].text.strip().lower()
                if label != "high":
                    continue

                # Forecast column is usually second numeric column in image 1
                forecast_col = cols[1]
                span = forecast_col.find_element(By.CSS_SELECTOR, "span.wu-value, span[class*='wu-value']")
                val = clean_temp_to_celsius(extract_number(span.text), unit)
                if val is not None:
                    return val
            except Exception:
                continue
    except Exception as e:
        print(f"    [{icao}] WARNING: WU row parse error: {e}")

    # Strategy 2: find the exact "High" row and grab first wu-value after it
    try:
        xpath = (
            "//div[contains(@class,'row') and contains(@class,'collapse')]"
            "[.//div[normalize-space()='High']]"
            "//span[contains(@class,'wu-value')]"
        )
        spans = driver.find_elements(By.XPATH, xpath)
        if spans:
            val = clean_temp_to_celsius(extract_number(spans[0].text), unit)
            if val is not None:
                return val
    except Exception:
        pass

    # Strategy 3: page-source regex fallback
    try:
        html = driver.page_source
        # Look for High row followed by wu-value-to text content
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

    print(f"    [{icao}] WARNING: could not parse Wunderground forecast high for {today}")
    return None


# ---------------------------------------------------------------------------
# S2 — Meteofor today's max
# ---------------------------------------------------------------------------

def fetch_meteofor_today_max_selenium(
    driver: webdriver.Chrome,
    airport: str,
    tz_name: str,
) -> float | None:
    path = METEOFOR_PATHS.get(airport)
    if not path:
        print(f"    [{airport}] WARNING: no Meteofor path configured")
        return None

    url = f"https://meteofor.com/{path}/"
    driver.get(url)

    try:
        WebDriverWait(driver, PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2.0)
        dismiss_common_popups(driver)
    except Exception:
        print(f"    [{airport}] WARNING: Meteofor body did not load")
        return None

    # Strategy 1: active weather tab contains <temperature-value value="21" ...>
    try:
        active_tab = driver.find_element(By.CSS_SELECTOR, ".weathertab.is-active, .weathertabs .is-active")
        temps = active_tab.find_elements(By.CSS_SELECTOR, "temperature-value[value]")
        vals = []
        for t in temps:
            raw = t.get_attribute("value")
            num = clean_temp_to_celsius(extract_number(raw), "C")
            if num is not None:
                vals.append(num)
        if vals:
            return round(max(vals), 3)
    except Exception:
        pass

    # Strategy 2: any visible today panel values
    try:
        elems = driver.find_elements(By.CSS_SELECTOR, "temperature-value[value]")
        vals = []
        for elem in elems:
            try:
                raw = elem.get_attribute("value")
                num = clean_temp_to_celsius(extract_number(raw), "C")
                if num is not None and -80 <= num <= 70:
                    vals.append(num)
            except Exception:
                continue
        if vals:
            # The page has hourly points + top tiles; max of visible values for today tab
            return round(max(vals), 3)
    except Exception:
        pass

    # Strategy 3: parse page source for temperature-value entries
    try:
        html = driver.page_source
        matches = re.findall(r'<temperature-value[^>]*value="(-?\d+(?:\.\d+)?)"', html, flags=re.IGNORECASE)
        vals = [clean_temp_to_celsius(x, "C") for x in matches]
        vals = [v for v in vals if v is not None and -80 <= v <= 70]
        if vals:
            return round(max(vals), 3)
    except Exception:
        pass

    print(f"    [{airport}] WARNING: could not parse Meteofor today max")
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
    tz_name = meta["tz"]
    icao = meta["icao"]

    driver = make_driver()
    _ = make_api_session()  # kept in case you want requests-based sources later

    try:
        pulled_at_local = now_local_naive(tz_name)
        forecast_date_local = today_local_str(tz_name)

        observed_max = load_observed_max_today(airport, tz_name)
        print(f"  [{airport}] Observed max today = {observed_max}")

        # S1 — Wunderground forecast almanac high
        src1 = None
        try:
            src1 = fetch_wunderground_forecast_high_selenium(driver, icao, tz_name)
        except Exception as e:
            print(f"  [{airport}] S1 Wunderground FAILED: {e}")

        # S2 — Meteofor today max
        src2 = None
        try:
            src2 = fetch_meteofor_today_max_selenium(driver, airport, tz_name)
        except Exception as e:
            print(f"  [{airport}] S2 Meteofor FAILED: {e}")

        # S3 — intentionally blank for now
        src3 = math.nan

        avg_val = mean_ignore_none([src1, src2, src3])

        print(
            f"  [{airport}] S1={src1}  S2={src2}  S3={src3}  AVG={avg_val}"
        )

        return {
            "airport":             airport,
            "icao":                icao,
            "forecast_date_local": forecast_date_local,
            "pulled_at_local":     pulled_at_local,
            "forecast_source_1":   src1,
            "forecast_source_2":   src2,
            "forecast_source_3":   src3,
            "forecast_avg_max":    avg_val,
            "observed_max_today":  observed_max,
            "source_1_name":       "wunderground_forecast_high",
            "source_2_name":       "meteofor_today_max",
            "source_3_name":       "nan",
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

    new_df = pd.DataFrame(rows)
    existing = load_existing(OUTPUT_PATH)
    existing = purge_old_forecasts(existing)
    combined = pd.concat([existing, new_df], ignore_index=True)

    combined["pulled_at_local"] = pd.to_datetime(combined["pulled_at_local"], errors="coerce")
    for col in ["forecast_source_1", "forecast_source_2", "forecast_source_3", "forecast_avg_max", "observed_max_today"]:
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
