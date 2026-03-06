"""
fetch_observations.py
─────────────────────
Scrapes hourly temperature observations from Weather Underground history pages
using Playwright (headless browser) for each airport over the last 3 days.

Install dependencies:
    pip install playwright pandas pyarrow
    playwright install chromium

Output: data/dashboard/observations.parquet
Columns: airport | timestamp_utc | timestamp_local | temp
"""

import os
import re
import time
import pandas as pd
from datetime import timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ── Airport config ─────────────────────────────────────────────────────────────
# URL pattern: https://www.wunderground.com/history/daily/{country}/{city}/{icao}/date/{YYYY-M-D}
STATIONS = {
    "ATL": {"url_path": "us/atlanta/KATL",         "tz": "America/New_York"},
    "NYC": {"url_path": "us/new-york-city/KLGA",   "tz": "America/New_York"},
    "CHI": {"url_path": "us/chicago/KORD",         "tz": "America/Chicago"},
    "DAL": {"url_path": "us/dallas/KDAL",          "tz": "America/Chicago"},
    "SEA": {"url_path": "us/seattle/KSEA",         "tz": "America/Los_Angeles"},
    "MIA": {"url_path": "us/miami/KMIA",           "tz": "America/New_York"},
    "TOR": {"url_path": "ca/toronto/CYYZ",         "tz": "America/Toronto"},
    "PAR": {"url_path": "fr/paris/LFPG",           "tz": "Europe/Paris"},
    "SEL": {"url_path": "kr/seoul/RKSI",           "tz": "Asia/Seoul"},
    "ANK": {"url_path": "tr/ankara/LTAC",          "tz": "Europe/Istanbul"},
    "BUE": {"url_path": "ar/buenos-aires/SAEZ",    "tz": "America/Argentina/Buenos_Aires"},
    "LON": {"url_path": "gb/london/EGLL",          "tz": "Europe/London"},
    "WLG": {"url_path": "nz/wellington/NZWN",      "tz": "Pacific/Auckland"},
}

BASE_URL = "https://www.wunderground.com/history/daily/{path}/date/{date}"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(DATA_DIR, "observations.parquet")

# ── Date range: today + previous 2 days ───────────────────────────────────────
NOW   = pd.Timestamp.now("UTC")
DATES = [(NOW - timedelta(days=i)).strftime("%Y-%-m-%-d") for i in range(2, -1, -1)]
# e.g. ["2026-3-5", "2026-3-6", "2026-3-7"]


def parse_temp(raw: str) -> float | None:
    """Extract numeric °C value from strings like '17 °C' or '62 °F'."""
    raw = raw.strip()
    m = re.search(r"([-\d.]+)\s*[°]?\s*([CF])?", raw)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "C").upper()
    if unit == "F":
        val = (val - 32) * 5 / 9
    return round(val, 1)


def parse_time_to_dt(time_str: str, date_str: str, tz: str) -> pd.Timestamp | None:
    """Parse '12:00 AM' + '2026-3-7' into a tz-aware timestamp."""
    try:
        # Normalise date to zero-padded format for strptime
        parts = date_str.split("-")
        norm_date = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        dt = pd.to_datetime(f"{norm_date} {time_str}", format="%Y-%m-%d %I:%M %p", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(f"{norm_date} {time_str}", errors="coerce")
        if pd.isna(dt):
            return None
        return dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    except Exception:
        return None


def scrape_day(page, airport: str, path: str, tz: str, date_str: str) -> pd.DataFrame:
    url = BASE_URL.format(path=path, date=date_str)
    print(f"    GET {url}")

    try:
        page.goto(url, wait_until="networkidle", timeout=30_000)
    except PWTimeout:
        print(f"    ✗ page load timeout")
        return pd.DataFrame()

    # Wait for the observations table — try multiple selectors
    table_found = False
    for selector in ["table.observation-table", "lib-city-history-observation table", "table"]:
        try:
            page.wait_for_selector(selector, timeout=12_000)
            table_found = True
            break
        except PWTimeout:
            continue

    if not table_found:
        print(f"    ✗ table not found (no data recorded?)")
        return pd.DataFrame()

    # Grab all table rows
    rows = page.query_selector_all("table tr")
    if not rows:
        print(f"    ✗ no rows found")
        return pd.DataFrame()

    records = []
    for row in rows[1:]:  # skip header
        cells = row.query_selector_all("td")
        if len(cells) < 2:
            continue
        time_str = cells[0].inner_text().strip()
        temp_str = cells[1].inner_text().strip()

        temp = parse_temp(temp_str)
        if temp is None:
            continue

        ts_local = parse_time_to_dt(time_str, date_str, tz)
        if ts_local is None:
            continue

        ts_utc = ts_local.tz_convert("UTC")

        records.append({
            "airport":         airport,
            "timestamp_utc":   ts_utc.tz_localize(None),
            "timestamp_local": ts_local.tz_localize(None),
            "temp":            temp,
        })

    df = pd.DataFrame(records)
    print(f"    ✓ {len(df)} rows")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
all_frames = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        locale="en-US",
    )
    page = context.new_page()
    # Block images/fonts to speed things up
    page.route("**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf,svg}", lambda r: r.abort())

    for airport, meta in STATIONS.items():
        print(f"\n── {airport} ──")
        airport_frames = []

        for date_str in DATES:
            df = scrape_day(page, airport, meta["url_path"], meta["tz"], date_str)
            if not df.empty:
                airport_frames.append(df)
            time.sleep(2)  # polite pacing between requests

        if airport_frames:
            combined = pd.concat(airport_frames, ignore_index=True)
            combined = combined.drop_duplicates(subset=["airport", "timestamp_local"])
            combined = combined.sort_values("timestamp_local").reset_index(drop=True)
            all_frames.append(combined)
            print(f"  → {len(combined)} total rows for {airport}")
        else:
            print(f"  → no data collected for {airport}")

    browser.close()

# ── Save ──────────────────────────────────────────────────────────────────────
if all_frames:
    final = pd.concat(all_frames, ignore_index=True)
    final = final.sort_values(["airport", "timestamp_local"]).reset_index(drop=True)
else:
    final = pd.DataFrame(columns=["airport", "timestamp_utc", "timestamp_local", "temp"])

final.to_parquet(OUTPUT_PATH, index=False)
print(f"\n✓ Saved {len(final):,} rows → {OUTPUT_PATH}")
print(f"  Airports : {sorted(final['airport'].unique().tolist())}")
if not final.empty:
    print(f"  Time range: {final['timestamp_local'].min()} → {final['timestamp_local'].max()}")
