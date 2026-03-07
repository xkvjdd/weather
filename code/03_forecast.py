# Requirements: requests, pandas, pyarrow, beautifulsoup4

import os
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

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

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(DATA_DIR, "forecast_latest.parquet")

HTTP_TIMEOUT  = 30
SLEEP_SECONDS = 0.5

HEADERS_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
HEADERS_API = {
    "User-Agent": "airport-weather-dashboard/1.0 (research; contact via github)"
}


def make_session(headers: dict) -> requests.Session:
    s = requests.Session()
    s.headers.update(headers)
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


# ---------------------------------------------------------------------------
# S1 — Wunderground almanac scrape
# URL: wunderground.com/history/daily/{ICAO}/date/{YYYY-MM-DD}
# Today's date is in the URL so can never bleed into tomorrow.
# The almanac table has rows: High / Low under Temperature.
# We find the table body row containing "High", then grab the FIRST
# td's wu-value-to span — that is the Forecast column value.
# Unit is detected from page context.
# ---------------------------------------------------------------------------

def fetch_wunderground_almanac_high(
    browser_session: requests.Session,
    icao: str,
    tz_name: str,
) -> float | None:
    today = today_local_str(tz_name)
    url   = f"https://www.wunderground.com/history/daily/{icao}/date/{today}"

    r = browser_session.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Detect unit — Wunderground serves °F for US, °C for international
    unit = "F"
    page_text = soup.get_text()
    if "°C" in page_text or "\u2103" in page_text:
        unit = "C"

    # Find the almanac section — it contains a table with High/Low rows
    # Navigate: find a <td> or <div> whose text is exactly "High", then
    # walk to its sibling/parent to get the forecast value span
    for tag in soup.find_all(string=re.compile(r"^\s*High\s*$")):
        parent = tag.parent
        if parent is None:
            continue
        # Look for wu-value-to span within the same row context
        row = parent.find_parent(["tr", "lib-city-history-almanac", "div"])
        if row is None:
            row = parent
        span = row.find("span", class_=re.compile(r"wu-value-to"))
        if span:
            try:
                val = float(span.get_text(strip=True))
                result = clean_temp_to_celsius(val, unit)
                if result is not None:
                    return result
            except ValueError:
                continue

    # Fallback: regex on raw HTML — find "High" label then first wu-value-to
    m = re.search(
        r">\s*High\s*<.{0,600}?wu-value[^>]*wu-value-to[^>]*>\s*(\-?\d+(?:\.\d+)?)\s*<",
        str(soup),
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        result = clean_temp_to_celsius(m.group(1), unit)
        if result is not None:
            return result

    print(f"    WARNING: Wunderground — could not parse almanac high for {icao} on {today}")
    return None


# ---------------------------------------------------------------------------
# S2 — Open-Meteo ECMWF IFS 0.25°
# Gold standard global NWP model, genuinely different from GFS.
# start_date=end_date=today (local) prevents cross-day bleed.
# ---------------------------------------------------------------------------

def fetch_open_meteo_ecmwf_daily_max(
    api_session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
) -> float | None:
    return _fetch_open_meteo_daily_max(api_session, lat, lon, tz_name, "ecmwf_ifs025")


# ---------------------------------------------------------------------------
# S3 — Open-Meteo regional best-fit model
# Uses the best available national NWP model for each airport's region.
# ---------------------------------------------------------------------------

def fetch_open_meteo_regional_daily_max(
    api_session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
    model: str,
) -> float | None:
    return _fetch_open_meteo_daily_max(api_session, lat, lon, tz_name, model)


def _fetch_open_meteo_daily_max(
    api_session: requests.Session,
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
    r     = api_session.get(url, params=params, timeout=HTTP_TIMEOUT)
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
# For each airport, drops any row whose forecast_date_local != today in
# that airport's own timezone. Keeps the parquet lean — today only.
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
    api_session: requests.Session,
    browser_session: requests.Session,
    airport: str,
    meta: dict,
) -> dict:
    lat, lon, tz_name, icao = meta["lat"], meta["lon"], meta["tz"], meta["icao"]
    s3_model = S3_MODEL[airport]

    pulled_at_local     = now_local_naive(tz_name)
    forecast_date_local = today_local_str(tz_name)

    src1 = src2 = src3 = None

    # S1 — Wunderground almanac scrape (location-specific, today's date in URL)
    try:
        src1 = fetch_wunderground_almanac_high(browser_session, icao, tz_name)
    except Exception as e:
        print(f"  S1 Wunderground FAILED: {e}")

    # S2 — ECMWF IFS 0.25° (global gold standard NWP)
    try:
        src2 = fetch_open_meteo_ecmwf_daily_max(api_session, lat, lon, tz_name)
    except Exception as e:
        print(f"  S2 ECMWF FAILED: {e}")

    # S3 — Regional best-fit NWP model
    try:
        src3 = fetch_open_meteo_regional_daily_max(api_session, lat, lon, tz_name, s3_model)
    except Exception as e:
        print(f"  S3 {s3_model} FAILED: {e}")

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
        "source_2_name":       "open_meteo_ecmwf_ifs025_daily_max",
        "source_3_name":       f"open_meteo_{s3_model}_daily_max",
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
    api_session     = make_session(HEADERS_API)
    browser_session = make_session(HEADERS_BROWSER)
    rows: list[dict] = []

    for airport, meta in AIRPORTS.items():
        s3_model = S3_MODEL[airport]
        print(
            f"Fetching {airport} ({meta['icao']}) "
            f"[S1=wunderground | S2=ecmwf_ifs025 | S3={s3_model}] "
            f"local date: {today_local_str(meta['tz'])}"
        )
        try:
            row = fetch_airport_forecasts(api_session, browser_session, airport, meta)
            rows.append(row)
            print(
                f"  S1 Wunderground = {row['forecast_source_1']}\n"
                f"  S2 ECMWF        = {row['forecast_source_2']}\n"
                f"  S3 {s3_model:<20} = {row['forecast_source_3']}\n"
                f"  AVG             = {row['forecast_avg_max']}"
            )
        except Exception as e:
            print(f"  FAILED: {e}")

        time.sleep(SLEEP_SECONDS)

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
