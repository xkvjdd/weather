import os
import re
import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

STATIONS = {
    "ATL": {"station_id": "KATL"},
    "NYC": {"station_id": "KLGA"},
    "CHI": {"station_id": "KORD"},
    "DAL": {"station_id": "KDAL"},
    "SEA": {"station_id": "KSEA"},
    "MIA": {"station_id": "KMIA"},
    "TOR": {"station_id": "CYYZ"},
    "PAR": {"station_id": "LFPG"},
    "SEL": {"station_id": "RKSI"},
    "ANK": {"station_id": "LTAC"},
    "BUE": {"station_id": "SAEZ"},
    "LON": {"station_id": "EGLL"},
    "WLG": {"station_id": "NZWN"},
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
    options.add_argument("--window-size=1800,2600")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def build_weather_url(station_id: str) -> str:
    return f"https://www.wunderground.com/weather/{station_id}"


def wait_for_weather_page(driver: webdriver.Chrome, timeout: int = 30) -> None:
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "p.timestamp"))
    )
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                '//div[contains(@class,"current-temp")]//span[contains(@class,"wu-unit-temperature")]',
            )
        )
    )


def parse_temp_to_celsius(temp_text: str) -> float | None:
    if temp_text is None:
        return None

    s = " ".join(str(temp_text).replace("\xa0", " ").split()).strip()

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CF])?", s, flags=re.IGNORECASE)
    if not m:
        return None

    value = float(m.group(1))
    unit = (m.group(2) or "C").upper()

    if unit == "F":
        return round((value - 32.0) * 5.0 / 9.0, 3)

    return value


def parse_timestamp_from_page(timestamp_text: str) -> pd.Timestamp | None:
    if timestamp_text is None:
        return None

    s = " ".join(str(timestamp_text).split())
    s = re.sub(r"^access_time\s*", "", s, flags=re.IGNORECASE).strip()

    m = re.search(
        r"(\d{1,2}:\d{2}\s*[AP]M)\s+[A-Z]{2,5}\s+on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        s,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    time_part = m.group(1).strip()
    date_part = m.group(2).strip()

    ts = pd.to_datetime(
        f"{date_part} {time_part}",
        format="%B %d, %Y %I:%M %p",
        errors="coerce",
    )
    if pd.isna(ts):
        return None
    return ts


def fetch_station_row(driver: webdriver.Chrome, airport: str, station_id: str) -> dict:
    url = build_weather_url(station_id)
    print(f"  loading {url}")

    driver.get(url)
    wait_for_weather_page(driver)

    timestamp_text = driver.find_element(By.CSS_SELECTOR, "p.timestamp").text
    temp_text = driver.find_element(
        By.XPATH,
        '//div[contains(@class,"current-temp")]//span[contains(@class,"wu-unit-temperature")]',
    ).text

    timestamp_local = parse_timestamp_from_page(timestamp_text)
    temp_c = parse_temp_to_celsius(temp_text)

    if timestamp_local is None:
        raise RuntimeError(f"could not parse local timestamp from: {timestamp_text}")

    if temp_c is None:
        raise RuntimeError(f"could not parse temp from: {temp_text}")

    return {
        "airport": airport,
        "timestamp_local": timestamp_local,
        "temp": temp_c,
    }


def load_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_parquet(path)
        keep = [c for c in ["airport", "timestamp_local", "temp"] if c in df.columns]
        if keep:
            return df[keep].copy()
    return pd.DataFrame(columns=["airport", "timestamp_local", "temp"])


def main() -> None:
    driver = make_driver()
    rows = []

    try:
        for airport, meta in STATIONS.items():
            print(f"Fetching {airport}")
            try:
                row = fetch_station_row(driver, airport, meta["station_id"])
                rows.append(row)
                print(f"  local={row['timestamp_local']} temp_c={row['temp']}")
            except Exception as e:
                print(f"  FAILED: {e}")

            time.sleep(1)

    finally:
        driver.quit()

    if not rows:
        print("No rows fetched; nothing saved.")
        return

    new_df = pd.DataFrame(rows, columns=["airport", "timestamp_local", "temp"])
    existing = load_existing(OUTPUT_PATH)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined["timestamp_local"] = pd.to_datetime(combined["timestamp_local"], errors="coerce")
    combined["temp"] = pd.to_numeric(combined["temp"], errors="coerce")

    combined = combined.dropna(subset=["airport", "timestamp_local", "temp"]).copy()

    combined["bucket_5m"] = combined["timestamp_local"].dt.floor("5min")
    combined = (
        combined.sort_values(["airport", "timestamp_local"])
        .drop_duplicates(subset=["airport", "bucket_5m"], keep="last")
        .drop(columns=["bucket_5m"])
        .sort_values(["airport", "timestamp_local"])
        .reset_index(drop=True)
    )

    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(combined):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
