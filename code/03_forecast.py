import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup


MAX_WORKERS = 16
TIMEOUT = 12

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

AIRPORT_META = {
    "ATL": {"name": "Hartsfield Jackson Atlanta International Airport", "tz": "America/New_York"},
    "NYC": {"name": "LaGuardia Airport", "tz": "America/New_York"},
    "CHI": {"name": "O'Hare International Airport", "tz": "America/Chicago"},
    "DAL": {"name": "Dallas Love Field", "tz": "America/Chicago"},
    "SEA": {"name": "Seattle Tacoma International Airport", "tz": "America/Los_Angeles"},
    "MIA": {"name": "Miami International Airport", "tz": "America/New_York"},
    "TOR": {"name": "Toronto Pearson International Airport", "tz": "America/Toronto"},
    "PAR": {"name": "Charles de Gaulle Airport", "tz": "Europe/Paris"},
    "SEL": {"name": "Incheon International Airport", "tz": "Asia/Seoul"},
    "ANK": {"name": "Esenboga Airport", "tz": "Europe/Istanbul"},
    "BUE": {"name": "Ezeiza International Airport", "tz": "America/Argentina/Buenos_Aires"},
    "LON": {"name": "Heathrow Airport", "tz": "Europe/London"},
    "WLG": {"name": "Wellington Airport", "tz": "Pacific/Auckland"},
}

# Fill with your final exact links
# S1 = AccuWeather
# S2 = BBC
# S3 = Weather.com
AIRPORT_LINKS = {
    "ATL": {
        "S1": "https://www.accuweather.com/en/us/atlanta/30303/weather-forecast/348181",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/9ec68ae9b6bf94371f1eab8b71aa9a0a83d7cb663bfaacc25cbaa51622e102c3",
    },
    "NYC": {"S1": "", "S2": "", "S3": ""},
    "CHI": {"S1": "", "S2": "", "S3": ""},
    "DAL": {"S1": "", "S2": "", "S3": ""},
    "SEA": {"S1": "", "S2": "", "S3": ""},
    "MIA": {"S1": "", "S2": "", "S3": ""},
    "TOR": {"S1": "", "S2": "", "S3": ""},
    "PAR": {"S1": "", "S2": "", "S3": ""},
    "SEL": {"S1": "", "S2": "", "S3": ""},
    "ANK": {"S1": "", "S2": "", "S3": ""},
    "BUE": {"S1": "", "S2": "", "S3": ""},
    "LON": {
        "S1": "https://www.accuweather.com/en/gb/london-heathrow-airport/tw6-2/weather-forecast/5346_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/92bcee466601fa8f3d322d6622fb528856f2fb4c2c12dbc52f91dd44e533ee5a",
    },
    "WLG": {"S1": "", "S2": "", "S3": ""},
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def now_local(code: str) -> datetime:
    return datetime.now(ZoneInfo(AIRPORT_META[code]["tz"]))


def c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def f_to_c(f: float) -> float:
    return round((f - 32) * 5 / 9, 1)


def first_int(text: str):
    if text is None:
        return None
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=1)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch_text(session: requests.Session, url: str):
    if not url:
        return None, "blank_url"
    try:
        r = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return r.text, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ============================================================
# S1 ACCUWEATHER
# FIX: take TODAY'S WEATHER card high, not 10-day list
# Screenshot target:
# div.today-forecast-card ... <p>Low clouds</p> <b>Hi: 12°</b>
# ============================================================

def parse_accuweather_today_high(code: str, html: str):
    soup = BeautifulSoup(html, "html.parser")

    # 1) Exact today card target
    cards = soup.select("div.today-forecast-card")
    for idx, card in enumerate(cards):
        b_tags = card.select("b")
        for b in b_tags:
            txt = clean_text(b.get_text(" ", strip=True))
            m = re.search(r"\bHi\s*:\s*(-?\d+)", txt, flags=re.I)
            if m:
                c = int(m.group(1))
                return {
                    "temp_c": c,
                    "temp_f": c_to_f(c),
                    "method": f"accuweather_today_card_b_{idx}",
                }

        card_text = clean_text(card.get_text(" ", strip=True))
        m = re.search(r"\bHi\s*:\s*(-?\d+)", card_text, flags=re.I)
        if m:
            c = int(m.group(1))
            return {
                "temp_c": c,
                "temp_f": c_to_f(c),
                "method": f"accuweather_today_card_text_{idx}",
            }

    # 2) Backup: look for TODAY'S WEATHER block in raw HTML
    raw_patterns = [
        r'today-forecast-card.*?\bHi\s*:\s*(-?\d+)',
        r"\bTODAY'S WEATHER\b.{0,500}?\bHi\s*:\s*(-?\d+)",
        r"\bHi\s*:\s*(-?\d+)\s*°",
    ]
    for i, patt in enumerate(raw_patterns, start=1):
        m = re.search(patt, html, flags=re.I | re.S)
        if m:
            c = int(m.group(1))
            return {
                "temp_c": c,
                "temp_f": c_to_f(c),
                "method": f"accuweather_regex_{i}",
            }

    return None


# ============================================================
# S2 BBC
# Leave as-is
# ============================================================

def parse_bbc_today_high(code: str, html: str):
    text = clean_text(html)
    dt = now_local(code)
    weekday_full = dt.strftime("%A")
    day_num = str(dt.day)

    patterns = [
        r'"maxTempC"\s*:\s*(-?\d+)',
        r'"maximumTemperature"\s*:\s*(-?\d+)',
        rf"\b{weekday_full}\b.*?\b{day_num}\b.*?\bHigh\b.*?(-?\d+)\s*[°º]C",
        r"\bHigh\b.*?(-?\d+)\s*[°º]C",
    ]

    for i, patt in enumerate(patterns, start=1):
        m = re.search(patt, html if i <= 2 else text, flags=re.I | re.S)
        if m:
            c = int(m.group(1))
            return {
                "temp_c": c,
                "temp_f": c_to_f(c),
                "method": f"bbc_p{i}",
            }

    return None


# ============================================================
# S3 WEATHER.COM
# FIX: take expanded Today DailyContent TemperatureValue
# Screenshot target:
# div[data-testid="DailyContent"] > span[data-testid="TemperatureValue"] = 11
# ============================================================

def parse_weathercom_today_high(code: str, html: str):
    soup = BeautifulSoup(html, "html.parser")

    # 1) Exact expanded Today block
    for idx, block in enumerate(soup.select('div[data-testid="DailyContent"]')):
        temp_span = block.select_one('span[data-testid="TemperatureValue"]')
        if temp_span:
            temp_val = first_int(temp_span.get_text(" ", strip=True))
            if temp_val is not None:
                return {
                    "temp_c": temp_val,   # your screenshot is in °C
                    "temp_f": c_to_f(temp_val),
                    "method": f"weathercom_dailycontent_{idx}",
                }

    # 2) Disclosure summary fallback
    for idx, summ in enumerate(soup.select('summary[data-testid="Disclosure-Summary"]')):
        temp_span = summ.select_one('span[data-testid="TemperatureValue"]')
        if temp_span:
            temp_val = first_int(temp_span.get_text(" ", strip=True))
            if temp_val is not None:
                return {
                    "temp_c": temp_val,
                    "temp_f": c_to_f(temp_val),
                    "method": f"weathercom_summary_{idx}",
                }

    # 3) Raw HTML fallback
    raw_patterns = [
        r'data-testid="DailyContent".{0,600}?data-testid="TemperatureValue"[^>]*>\s*(-?\d+)',
        r'data-testid="TemperatureValue"[^>]*>\s*(-?\d+)\s*<',
        r'\bToday\b.{0,300}?(-?\d+)\s*°',
    ]
    for i, patt in enumerate(raw_patterns, start=1):
        m = re.search(patt, html, flags=re.I | re.S)
        if m:
            c = int(m.group(1))
            return {
                "temp_c": c,
                "temp_f": c_to_f(c),
                "method": f"weathercom_regex_{i}",
            }

    return None


def parse_source(code: str, source: str, html: str):
    if source == "S1":
        return parse_accuweather_today_high(code, html)
    if source == "S2":
        return parse_bbc_today_high(code, html)
    if source == "S3":
        return parse_weathercom_today_high(code, html)
    return None


def fetch_one(session: requests.Session, code: str, source: str, url: str):
    html, err = fetch_text(session, url)
    if err:
        return {
            "code": code,
            "source": source,
            "url": url,
            "ok": False,
            "error": err,
            "temp_c": None,
            "temp_f": None,
            "method": None,
        }

    parsed = parse_source(code, source, html)
    if not parsed:
        return {
            "code": code,
            "source": source,
            "url": url,
            "ok": False,
            "error": "parse_failed",
            "temp_c": None,
            "temp_f": None,
            "method": None,
        }

    return {
        "code": code,
        "source": source,
        "url": url,
        "ok": True,
        "error": None,
        "temp_c": parsed["temp_c"],
        "temp_f": parsed["temp_f"],
        "method": parsed["method"],
    }


def process_all():
    results = {
        code: {
            "code": code,
            "airport": AIRPORT_META[code]["name"],
            "local_now": now_local(code).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "S1_url": AIRPORT_LINKS[code]["S1"],
            "S2_url": AIRPORT_LINKS[code]["S2"],
            "S3_url": AIRPORT_LINKS[code]["S3"],
            "S1_temp_c": None,
            "S1_temp_f": None,
            "S1_method": None,
            "S1_error": None,
            "S2_temp_c": None,
            "S2_temp_f": None,
            "S2_method": None,
            "S2_error": None,
            "S3_temp_c": None,
            "S3_temp_f": None,
            "S3_method": None,
            "S3_error": None,
        }
        for code in AIRPORT_META
    }

    jobs = []
    with make_session() as session:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for code, links in AIRPORT_LINKS.items():
                for source in ("S1", "S2", "S3"):
                    url = links.get(source, "")
                    if url:
                        jobs.append(ex.submit(fetch_one, session, code, source, url))
                    else:
                        results[code][f"{source}_error"] = "blank_url"

            for fut in as_completed(jobs):
                row = fut.result()
                code = row["code"]
                source = row["source"]

                results[code][f"{source}_error"] = row["error"]
                results[code][f"{source}_temp_c"] = row["temp_c"]
                results[code][f"{source}_temp_f"] = row["temp_f"]
                results[code][f"{source}_method"] = row["method"]

    return list(results.values())


def main():
    rows = process_all()

    for row in rows:
        print(f'\n=== {row["code"]} | {row["airport"]} ===')
        print("Local now:", row["local_now"])
        print("S1:", row["S1_temp_c"], "C |", row["S1_temp_f"], "F |", row["S1_method"], "|", row["S1_error"])
        print("S2:", row["S2_temp_c"], "C |", row["S2_temp_f"], "F |", row["S2_method"], "|", row["S2_error"])
        print("S3:", row["S3_temp_c"], "C |", row["S3_temp_f"], "F |", row["S3_method"], "|", row["S3_error"])

    print("\nFINAL_JSON =")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
