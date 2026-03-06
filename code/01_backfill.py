import os
import time
from datetime import datetime, timedelta

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

STATIONS = {
    "ATL": {"icao": "KATL", "history_path": "us/ga/atlanta/KATL"},
    "NYC": {"icao": "KLGA", "history_path": "us/ny/new-york/KLGA"},
    "CHI": {"icao": "KORD", "history_path": "us/il/chicago/KORD"},
    "DAL": {"icao": "KDAL", "history_path": "us/tx/dallas/KDAL"},
    "SEA": {"icao": "KSEA", "history_path": "us/wa/seattle/KSEA"},
    "MIA": {"icao": "KMIA", "history_path": "us/fl/miami/KMIA"},
    "TOR": {"icao": "CYYZ", "history_path": "ca/on/toronto/CYYZ"},
    "PAR": {"icao": "LFPG", "history_path": "fr/paris/LFPG"},
    "SEL": {"icao": "RKSI", "history_path": "kr/incheon/RKSI"},
    "ANK": {"icao": "LTAC", "history_path": "tr/ankara/LTAC"},
    "BUE": {"icao": "SAEZ", "history_path": "ar/buenos-aires/SAEZ"},
    "LON": {"icao": "EGLL", "history_path": "gb/england/london/EGLL"},
    "WLG": {"icao": "NZWN", "history_path": "nz/wellington/NZWN"},
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "observations.parquet")


def make_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1600,2200")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def build_history_url(history_path: str, dt_local: pd.Timestamp) -> str:
    date_str = f"{dt_local.year}-{dt_local.month}-{dt_local.day}"
    return f"https://www.wunderground.com/history/daily/{history_path}/date/{date_str}"


def wait_for_table(driver: webdriver.Chrome, timeout: int = 25) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        html = driver.page_source
        if "Daily Observations" in html and "<table" in html:
            return True
        time.sleep(1)
    return False


def parse_temp_c(value) -> float | None:
    if pd.isna(value):
        return None
    s = str(value).replace("°C", "").replace("°", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def parse_time_to_timestamp_local(date_local: pd.Timestamp, time_str: str) -> pd.Timestamp | None:
    try:
        dt = pd.to_datetime(f"{date_local.strftime('%Y-%m-%d')} {time_str}", format="%Y-%m-%d %I:%M %p")
        return dt
    except Exception:
        try:
            return pd.to_datetime(f"{date_local.strftime('%Y-%m-%d')} {time_str}")
        except Exception:
            return None


def scrape_daily_observations(driver: webdriver.Chrome, airport: str, dt_local: pd.Timestamp) -> pd.DataFrame:
    history_path = STATIONS[airport]["history_path"]
    url = build_history_url(history_path, dt_local)

    print(f"  loading {url}")
    driver.get(url)

    ok = wait_for_table(driver)
    if not ok:
        raise RuntimeError("daily observations table did not render")

    tables = pd.read_html(driver.page_source)
    if not tables:
        raise RuntimeError("no html tables found after render")

    obs = None
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "Time" in cols and "Temperature" in cols:
            obs = t.copy()
            break

    if obs is None:
        raise RuntimeError("could not find Daily Observations table")

    obs.columns = [str(c).strip() for c in obs.columns]

    obs["timestamp_local"] = obs["Time"].apply(lambda x: parse_time_to_timestamp_local(dt_local, x))
    obs = obs[obs["timestamp_local"].notna()].copy()

    obs["temp"] = obs["Temperature"].apply(parse_temp_c)
    obs = obs[obs["temp"].notna()].copy()

    keep_cols = ["timestamp_local", "temp"]
    extra_cols = [c for c in ["Dew Point", "Humidity", "Wind", "Wind Speed", "Wind Gust", "Pressure", "Precip.", "Condition"] if c in obs.columns]
    out = obs[keep_cols + extra_cols].copy()

    out.insert(0, "airport", airport)
    out.insert(1, "source", "wunderground_history")
    out["date_local"] = dt_local.normalize()

    return out.sort_values("timestamp_local").reset_index(drop=True)


def add_timestamp_utc(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for airport, g in df.groupby("airport", sort=False):
        tz_name = CITY_TO_TZ[airport]
        x = g.copy()
        ts_local = pd.to_datetime(x["timestamp_local"], errors="coerce")
        ts_utc = ts_local.dt.tz_localize(tz_name, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
        x["timestamp_utc"] = ts_utc.dt.tz_localize(None)
        pieces.append(x)

    out = pd.concat(pieces, ignore_index=True)
    return out


def main():
    driver = make_driver()
    frames = []

    try:
        for airport in STATIONS:
            print(f"Downloading {airport}")

            tz_name = CITY_TO_TZ[airport]
            now_local = pd.Timestamp.now(tz=tz_name)

            for d in range(0, 3):
                dt_local = (now_local.normalize() - pd.Timedelta(days=d))
                try:
                    day_df = scrape_daily_observations(driver, airport, dt_local)
                    if day_df.empty:
                        print(f"  {dt_local.date()}: no rows")
                        continue

                    frames.append(day_df)
                    print(f"  {dt_local.date()}: kept {len(day_df):,} rows")
                except Exception as e:
                    print(f"  {dt_local.date()}: FAILED: {e}")

    finally:
        driver.quit()

    if frames:
        final = pd.concat(frames, ignore_index=True)
        final = add_timestamp_utc(final)
        final = final.sort_values(["airport", "timestamp_local"]).reset_index(drop=True)

        final = final.drop_duplicates(subset=["airport", "timestamp_local"], keep="last").reset_index(drop=True)

        ordered = ["airport", "source", "date_local", "timestamp_local", "timestamp_utc", "temp"]
        others = [c for c in final.columns if c not in ordered]
        final = final[ordered + others]
    else:
        final = pd.DataFrame(
            columns=["airport", "source", "date_local", "timestamp_local", "timestamp_utc", "temp"]
        )

    final.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(final):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
