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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(DATA_DIR, "forecast_latest.parquet")

HTTP_TIMEOUT  = 30
SLEEP_SECONDS = 0.5

# Wunderground needs a browser-like UA; aviationweather.gov wants a descriptive one
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
# S1 — aviationweather.gov TAF
# Free, no key, global ICAO coverage.
# TAFs contain TX (max temp) and TN (min temp) fields in the forecast periods
# for many international stations. We parse all forecast periods that fall
# within today (local) and return the max temperature seen.
# Note: not all TAFs include TX/TN (US domestic TAFs often omit them), so
# this may return None for some airports — that's expected and the avg
# gracefully handles it.
# ---------------------------------------------------------------------------

def fetch_taf_max_today(
    api_session: requests.Session,
    icao: str,
    tz_name: str,
) -> float | None:
    url = f"https://aviationweather.gov/api/data/taf"
    params = {"ids": icao, "format": "json", "metar": "false"}
    r = api_session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()

    data = r.json()
    if not data:
        print(f"    WARNING: TAF — empty response for {icao}")
        return None

    taf = data[0] if isinstance(data, list) else data
    today = today_local_str(tz_name)
    tz    = ZoneInfo(tz_name)

    temps: list[float] = []

    # TAF forecast periods — look for temperature fields
    for period in taf.get("fcsts", []):
        # Check period start falls within today local
        valid_time = period.get("timeFrom") or period.get("validTimeFrom")
        if valid_time:
            try:
                # aviationweather returns epoch seconds or ISO string
                if isinstance(valid_time, (int, float)):
                    dt_local = datetime.fromtimestamp(valid_time, tz=timezone.utc).astimezone(tz)
                else:
                    dt_local = datetime.fromisoformat(valid_time).astimezone(tz)
                if dt_local.date().isoformat() != today:
                    continue
            except Exception:
                pass  # if we can't parse time, still check for temp

        # TX field (forecast max temp, Celsius)
        for key in ("maxTemp", "tx", "tempMax", "temperature"):
            val = period.get(key)
            if val is not None:
                c = clean_temp_to_celsius(val, "C")
                if c is not None:
                    temps.append(c)

    if temps:
        return round(max(temps), 3)

    print(f"    WARNING: TAF — no temperature fields found for {icao} on {today}")
    return None


# ---------------------------------------------------------------------------
# S2 — Wunderground almanac scrape
# Targets: https://www.wunderground.com/history/daily/{ICAO}/date/{YYYY-MM-DD}
# The almanac table for today's date contains a "Forecast" column with the
# high temp. The value lives in:
#   span.wu-value.wu-value-to   (inside the High row of the temperature table)
# Unit is shown separately — page defaults to °F for US, °C for international,
# but we detect from the label and always convert to Celsius.
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

    # Find all wu-value spans — the almanac forecast high is the first
    # wu-value.wu-value-to inside the temperature section "High" row.
    # We locate the "High" label row then grab the first wu-value-to within it.

    # Strategy: find all rows, look for one whose text contains "High"
    forecast_high: float | None = None
    unit = "F"  # Wunderground defaults; we'll detect below

    # Detect unit from page — look for °C indicator anywhere in almanac table
    page_text = soup.get_text()
    if "°C" in page_text or "&#8451;" in r.text:
        unit = "C"

    # Find the almanac section specifically (avoid picking up current conditions)
    # The almanac table has class "history-table" or similar; look for the
    # forecast High value via the wu-value-to span pattern
    tables = soup.find_all("lib-city-history-almanac") or soup.find_all(
        "div", class_=re.compile(r"history|almanac", re.I)
    )

    # Fallback: just grab ALL wu-value-to spans and take the first numeric one
    # that appears after a "High" label in the DOM order
    all_spans = soup.find_all("span", class_=re.compile(r"wu-value-to"))

    # Walk the DOM looking for a "High" text node followed by a wu-value-to span
    body_text = str(soup)
    # Pattern: "High" ... wu-value-to">NUMBER<
    m = re.search(
        r"High\b.{0,400}?wu-value[^>]*wu-value-to[^>]*>(\-?\d+(?:\.\d+)?)<",
        body_text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        forecast_high = clean_temp_to_celsius(m.group(1), unit)

    if forecast_high is None and all_spans:
        for span in all_spans:
            try:
                v = float(span.get_text(strip=True))
                forecast_high = clean_temp_to_celsius(v, unit)
                break
            except ValueError:
                continue

    if forecast_high is None:
        print(f"    WARNING: Wunderground almanac — could not parse high for {icao} on {today}")

    return forecast_high


# ---------------------------------------------------------------------------
# S3 — Open-Meteo GFS seamless daily max
# Grid-interpolated but fully automated, reliable, no key needed.
# start_date=end_date=today (local) prevents any cross-day bleed.
# ---------------------------------------------------------------------------

def fetch_open_meteo_gfs_daily_max(
    api_session: requests.Session,
    lat: float,
    lon: float,
    tz_name: str,
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
        "models":           "gfs_seamless",
    }
    r = api_session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data  = r.json()
    dates = data.get("daily", {}).get("time", [])
    vals  = data.get("daily", {}).get("temperature_2m_max", [])

    for d, v in zip(dates, vals):
        if d == today and v is not None:
            return clean_temp_to_celsius(v, "C")

    print(f"    WARNING: Open-Meteo GFS — no match for {today}, got: {dates}")
    return None


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

    pulled_at_local     = now_local_naive(tz_name)
    forecast_date_local = today_local_str(tz_name)

    src1 = src2 = src3 = None

    # S1 — TAF via aviationweather.gov
    try:
        src1 = fetch_taf_max_today(api_session, icao, tz_name)
    except Exception as e:
        print(f"  S1 TAF FAILED: {e}")

    # S2 — Wunderground almanac scrape
    try:
        src2 = fetch_wunderground_almanac_high(browser_session, icao, tz_name)
    except Exception as e:
        print(f"  S2 Wunderground FAILED: {e}")

    # S3 — Open-Meteo GFS
    try:
        src3 = fetch_open_meteo_gfs_daily_max(api_session, lat, lon, tz_name)
    except Exception as e:
        print(f"  S3 GFS FAILED: {e}")

    return {
        "airport":              airport,
        "icao":                 icao,
        "forecast_date_local":  forecast_date_local,
        "pulled_at_local":      pulled_at_local,
        "forecast_source_1":    src1,
        "forecast_source_2":    src2,
        "forecast_source_3":    src3,
        "forecast_avg_max":     mean_ignore_none([src1, src2, src3]),
        "source_1_name":        "aviationweather_taf_max",
        "source_2_name":        "wunderground_almanac_forecast_high",
        "source_3_name":        "open_meteo_gfs_seamless_daily_max",
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
        print(
            f"Fetching {airport} ({meta['icao']}) "
            f"local date: {today_local_str(meta['tz'])}"
        )
        try:
            row = fetch_airport_forecasts(api_session, browser_session, airport, meta)
            rows.append(row)
            print(
                f"  S1 TAF       = {row['forecast_source_1']}\n"
                f"  S2 Wunder    = {row['forecast_source_2']}\n"
                f"  S3 GFS       = {row['forecast_source_3']}\n"
                f"  AVG          = {row['forecast_avg_max']}"
            )
        except Exception as e:
            print(f"  FAILED: {e}")

        time.sleep(SLEEP_SECONDS)

    if not rows:
        print("No rows fetched; nothing saved.")
        return

    new_df   = pd.DataFrame(rows)
    existing = load_existing(OUTPUT_PATH)
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
