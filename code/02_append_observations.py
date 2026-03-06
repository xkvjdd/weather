import os
import re
import html
import pandas as pd
import requests

STATIONS = {
    "ATL": {"icao": "KATL"},
    "NYC": {"icao": "KLGA"},
    "CHI": {"icao": "KORD"},
    "DAL": {"icao": "KDAL"},
    "SEA": {"icao": "KSEA"},
    "MIA": {"icao": "KMIA"},
    "TOR": {"icao": "CYYZ"},
    "PAR": {"icao": "LFPG"},
    "SEL": {"icao": "RKSI"},
    "ANK": {"icao": "LTAC"},
    "BUE": {"icao": "SAEZ"},
    "LON": {"icao": "EGLL"},
    "WLG": {"icao": "NZWN"},
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

BASE_URL = "https://www.wunderground.com/weather/{icao}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
TIMEOUT = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "dashboard")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "observations_live.parquet")


def f_to_c(temp_f: float) -> float:
    return (temp_f - 32.0) * 5.0 / 9.0


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", "\n", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def extract_temp_f(text: str) -> float | None:
    patterns = [
        r"Updated just now.*?(?:\d+\s*°\s*\|\s*\d+\s*°).*?(\d+(?:\.\d+)?)\s*°F",
        r"Updated.*?(?:\d+\s*°\s*\|\s*\d+\s*°).*?(\d+(?:\.\d+)?)\s*°F",
        r"Weather Conditions.*?(\d+(?:\.\d+)?)\s*°F",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return float(m.group(1))
    return None


def extract_condition(text: str) -> str | None:
    patterns = [
        r"Updated just now.*?\d+\s*°F.*?\n([A-Za-z/\- ]{3,40})\n",
        r"Updated.*?\d+\s*°F.*?\n([A-Za-z/\- ]{3,40})\n",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(1).strip()
            if "Radar" not in val and "Satellite" not in val:
                return val
    return None


def fetch_station_row(airport: str, icao: str) -> dict | None:
    url = BASE_URL.format(icao=icao)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()

    text = html_to_text(resp.text)

    temp_f = extract_temp_f(text)
    if temp_f is None:
        raise ValueError(f"could not parse temp from {url}")

    condition = extract_condition(text)

    scraped_at_utc = pd.Timestamp.now("UTC")
    tz_name = CITY_TO_TZ[airport]
    scraped_at_local = scraped_at_utc.tz_convert(tz_name)

    return {
        "airport": airport,
        "icao": icao,
        "source": "wunderground",
        "scraped_at_utc": scraped_at_utc.tz_localize(None),
        "scraped_at_local": scraped_at_local.tz_localize(None),
        "temp_f": float(temp_f),
        "temp_c": round(f_to_c(float(temp_f)), 3),
        "condition": condition,
        "url": url,
    }


def load_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=[
        "airport", "icao", "source",
        "scraped_at_utc", "scraped_at_local",
        "temp_f", "temp_c", "condition", "url"
    ])


def main():
    rows = []

    for airport, meta in STATIONS.items():
        icao = meta["icao"]
        print(f"Fetching {airport} ({icao})")

        try:
            row = fetch_station_row(airport, icao)
            rows.append(row)
            print(f"  temp_f={row['temp_f']} temp_c={row['temp_c']}")
        except Exception as e:
            print(f"  FAILED: {e}")

    if not rows:
        print("No rows fetched; nothing to save.")
        return

    new_df = pd.DataFrame(rows)
    existing = load_existing(OUTPUT_PATH)

    combined = pd.concat([existing, new_df], ignore_index=True)

    combined["scraped_at_utc"] = pd.to_datetime(combined["scraped_at_utc"], errors="coerce")
    combined["scraped_at_local"] = pd.to_datetime(combined["scraped_at_local"], errors="coerce")

    # avoid accidental dupes if workflow re-runs in the same minute
    combined["bucket_5m"] = combined["scraped_at_utc"].dt.floor("5min")
    combined = (
        combined.sort_values(["airport", "scraped_at_utc"])
        .drop_duplicates(subset=["airport", "bucket_5m"], keep="last")
        .drop(columns=["bucket_5m"])
        .reset_index(drop=True)
    )

    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(combined):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
