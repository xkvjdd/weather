import os
import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

STATIONS = {
    "ATL": {"history_path": "us/ga/atlanta/KATL"},
    "NYC": {"history_path": "us/ny/new-york/KLGA"},
    "CHI": {"history_path": "us/il/chicago/KORD"},
    "DAL": {"history_path": "us/tx/dallas/KDAL"},
    "SEA": {"history_path": "us/wa/seattle/KSEA"},
    "MIA": {"history_path": "us/fl/miami/KMIA"},
    "TOR": {"history_path": "ca/on/toronto/CYYZ"},
    "PAR": {"history_path": "fr/paris/LFPG"},
    "SEL": {"history_path": "kr/incheon/RKSI"},
    "ANK": {"history_path": "tr/ankara/LTAC"},
    "BUE": {"history_path": "ar/buenos-aires/SAEZ"},
    "LON": {"history_path": "gb/england/london/EGLL"},
    "WLG": {"history_path": "nz/wellington/NZWN"},
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


def build_history_url(history_path: str, dt_local: pd.Timestamp) -> str:
    return (
        f"https://www.wunderground.com/history/daily/"
        f"{history_path}/date/{dt_local.year}-{dt_local.month}-{dt_local.day}"
    )


def wait_for_table(driver: webdriver.Chrome, timeout: int = 30) -> None:
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "tbody tr"))
    )
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "td.mat-column-dateString"))
    )
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "td.mat-column-temperature"))
    )


def parse_temp_c(temp_text: str) -> float | None:
    if temp_text is None:
        return None
    s = str(temp_text).replace("°C", "").replace("°", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def parse_local_timestamp(date_local: pd.Timestamp, time_text: str) -> pd.Timestamp | None:
    if time_text is None:
        return None

    s = " ".join(str(time_text).split())
    candidate = f"{date_local.strftime('%Y-%m-%d')} {s}"

    fmts = [
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %H:%M",
    ]

    for fmt in fmts:
        try:
            return pd.to_datetime(candidate, format=fmt)
        except Exception:
            pass

    ts = pd.to_datetime(candidate, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def scrape_one_day(driver: webdriver.Chrome, airport: str, dt_local: pd.Timestamp) -> pd.DataFrame:
    url = build_history_url(STATIONS[airport]["history_path"], dt_local)
    print(f"  loading {url}")

    driver.get(url)
    wait_for_table(driver)

    rows = driver.find_elements(By.CSS_SELECTOR, "tbody tr")
    data = []

    for row in rows:
        try:
            time_cell = row.find_element(By.CSS_SELECTOR, "td.mat-column-dateString").text
            temp_cell = row.find_element(By.CSS_SELECTOR, "td.mat-column-temperature").text

            ts_local = parse_local_timestamp(dt_local, time_cell)
            temp = parse_temp_c(temp_cell)

            if ts_local is None or temp is None:
                continue

            data.append({
                "airport": airport,
                "timestamp_local": ts_local,
                "temp": temp,
            })

        except Exception:
            continue

    out = pd.DataFrame(data, columns=["airport", "timestamp_local", "temp"])
    if not out.empty:
        out = (
            out.dropna(subset=["airport", "timestamp_local", "temp"])
               .drop_duplicates(subset=["airport", "timestamp_local"], keep="last")
               .sort_values("timestamp_local")
               .reset_index(drop=True)
        )

    return out


def main() -> None:
    driver = make_driver()
    frames = []

    try:
        today_local_anchor = pd.Timestamp.now("UTC").tz_localize(None).normalize()

        for airport in STATIONS:
            print(f"Downloading {airport}")

            for d in range(0, 3):
                dt_local = today_local_anchor - pd.Timedelta(days=d)

                try:
                    day_df = scrape_one_day(driver, airport, dt_local)
                    if day_df.empty:
                        print(f"  {dt_local.date()}: no rows")
                        continue

                    frames.append(day_df)
                    print(f"  {dt_local.date()}: kept {len(day_df):,} rows")

                except Exception as e:
                    print(f"  {dt_local.date()}: FAILED: {e}")

                time.sleep(1)

    finally:
        driver.quit()

    if frames:
        final = pd.concat(frames, ignore_index=True)
        final = (
            final.drop_duplicates(subset=["airport", "timestamp_local"], keep="last")
                 .sort_values(["airport", "timestamp_local"])
                 .reset_index(drop=True)
        )
    else:
        final = pd.DataFrame(columns=["airport", "timestamp_local", "temp"])

    final.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(final):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
