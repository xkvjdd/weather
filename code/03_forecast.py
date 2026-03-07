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

# Fill S2 BBC links when you have them.
# S1 = AccuWeather
# S2 = BBC
# S3 = Weather.com
AIRPORT_LINKS = {
    "ATL": {
        "S1": "https://www.accuweather.com/en/us/hartsfield-jackson-atlanta-international-airport/30320/weather-forecast/9416_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Hartsfield%2BJackson%2BAtlanta%2BInt%27l%2BAirport%2BGA%2BATL%3A9%3AUS",
    },
    "NYC": {
        "S1": "https://www.accuweather.com/en/us/la-guardia-airport/11371/weather-forecast/26272_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/LaGuardia%2BAirport%2BNY",
    },
    "CHI": {
        "S1": "https://www.accuweather.com/en/us/chicago-ohare-international-airport/60666/weather-forecast/72530_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/OHare%2BInternational%2BAirport",
    },
    "DAL": {
        "S1": "https://www.accuweather.com/en/us/dallas-love-field/75235/weather-forecast/35119_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Dallas%2BLove%2BField",
    },
    "SEA": {
        "S1": "https://www.accuweather.com/en/us/seattle-tacoma-international-airport/98158/weather-forecast/351409_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Seattle%2BTacoma%2BIntl%2BAirport",
    },
    "MIA": {
        "S1": "https://www.accuweather.com/en/us/miami-international-airport/33122/weather-forecast/348308_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Miami%2BInternational%2BAirport",
    },
    "TOR": {
        "S1": "https://www.accuweather.com/en/ca/toronto-pearson-international-airport/l5p/weather-forecast/35185_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Toronto%2BPearson%2BIntl%2BAirport",
    },
    "PAR": {
        "S1": "https://www.accuweather.com/en/fr/roissy-en-france/135713/weather-forecast/135713",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Charles%2BDe%2BGaulle%2BAirport",
    },
    "SEL": {
        "S1": "https://www.accuweather.com/en/kr/incheon/226081/weather-forecast/226081",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Incheon%2BInternational%2BAirport",
    },
    "ANK": {
        "S1": "https://www.accuweather.com/en/tr/esenboga-havalimani/318795/weather-forecast/318795",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Esenboga%2BAirport",
    },
    "BUE": {
        "S1": "https://www.accuweather.com/en/ar/ezeiza/7894/weather-forecast/7894",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Ezeiza%2BInternational%2BAirport",
    },
    "LON": {
        "S1": "https://www.accuweather.com/en/gb/heathrow/328328/weather-forecast/328328",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Heathrow%2BAirport",
    },
    "WLG": {
        "S1": "https://www.accuweather.com/en/nz/wellington/250938/weather-forecast/250938",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/Wellington%2BWellington%2B6011%2BNew%2BZealand",
    },
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
        return None, None, "blank_url"
    try:
        r = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return r.text, r.url, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


# ============================================================
# S1 ACCUWEATHER
# Target: TODAY'S WEATHER card -> Hi: xx°
# ============================================================

def parse_accuweather_today_high(code: str, html: str):
    soup = BeautifulSoup(html, "html.parser")

    # 1) Exact card selector
    for idx, card in enumerate(soup.select("div.today-forecast-card")):
        for b in card.select("b"):
            txt = clean_text(b.get_text(" ", strip=True))
            m = re.search(r"\bHi\s*:\s*(-?\d+)", txt, flags=re.I)
            if m:
                c = int(m.group(1))
                return {
                    "temp_c": c,
                    "temp_f": c_to_f(c),
                    "method": f"accuweather_today_card_b_{idx}",
                    "match_text": txt,
                }

        card_text = clean_text(card.get_text(" ", strip=True))
        m = re.search(r"\bHi\s*:\s*(-?\d+)", card_text, flags=re.I)
        if m:
            c = int(m.group(1))
            return {
                "temp_c": c,
                "temp_f": c_to_f(c),
                "method": f"accuweather_today_card_text_{idx}",
                "match_text": card_text[:300],
            }

    # 2) Raw HTML fallback
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
                "match_text": m.group(0)[:300],
            }

    return None


# ============================================================
# S2 BBC
# ============================================================

def parse_bbc_today_high(code: str, html: str):
    text = clean_text(html)
    dt = now_local(code)
    weekday_full = dt.strftime("%A")
    day_num = str(dt.day)

    patterns = [
        (r'"maxTempC"\s*:\s*(-?\d+)', html),
        (r'"maximumTemperature"\s*:\s*(-?\d+)', html),
        (rf"\b{weekday_full}\b.*?\b{day_num}\b.*?\bHigh\b.*?(-?\d+)\s*[°º]C", text),
        (r"\bHigh\b.*?(-?\d+)\s*[°º]C", text),
    ]

    for i, (patt, source_text) in enumerate(patterns, start=1):
        m = re.search(patt, source_text, flags=re.I | re.S)
        if m:
            c = int(m.group(1))
            return {
                "temp_c": c,
                "temp_f": c_to_f(c),
                "method": f"bbc_p{i}",
                "match_text": m.group(0)[:300],
            }

    return None


# ============================================================
# S3 WEATHER.COM
# Target: div[data-testid="DailyContent"] + TemperatureValue
# ============================================================

def parse_weathercom_today_high(code: str, html: str):
    soup = BeautifulSoup(html, "html.parser")

    # 1) Expanded today block
    for idx, block in enumerate(soup.select('div[data-testid="DailyContent"]')):
        temp_span = block.select_one('span[data-testid="TemperatureValue"]')
        if temp_span:
            temp_val = first_int(temp_span.get_text(" ", strip=True))
            if temp_val is not None:
                block_text = clean_text(block.get_text(" ", strip=True))
                return {
                    "temp_c": temp_val,  # assumes weather.com page currently in Celsius
                    "temp_f": c_to_f(temp_val),
                    "method": f"weathercom_dailycontent_{idx}",
                    "match_text": block_text[:300],
                }

    # 2) Summary fallback
    for idx, summ in enumerate(soup.select('summary[data-testid="Disclosure-Summary"]')):
        temp_span = summ.select_one('span[data-testid="TemperatureValue"]')
        if temp_span:
            temp_val = first_int(temp_span.get_text(" ", strip=True))
            if temp_val is not None:
                summ_text = clean_text(summ.get_text(" ", strip=True))
                return {
                    "temp_c": temp_val,
                    "temp_f": c_to_f(temp_val),
                    "method": f"weathercom_summary_{idx}",
                    "match_text": summ_text[:300],
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
                "match_text": m.group(0)[:300],
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
    html, final_url, err = fetch_text(session, url)
    if err:
        return {
            "code": code,
            "source": source,
            "url": url,
            "final_url": final_url,
            "ok": False,
            "error": err,
            "temp_c": None,
            "temp_f": None,
            "method": None,
            "match_text": None,
        }

    parsed = parse_source(code, source, html)
    if not parsed:
        return {
            "code": code,
            "source": source,
            "url": url,
            "final_url": final_url,
            "ok": False,
            "error": "parse_failed",
            "temp_c": None,
            "temp_f": None,
            "method": None,
            "match_text": None,
        }

    return {
        "code": code,
        "source": source,
        "url": url,
        "final_url": final_url,
        "ok": True,
        "error": None,
        "temp_c": parsed["temp_c"],
        "temp_f": parsed["temp_f"],
        "method": parsed["method"],
        "match_text": parsed.get("match_text"),
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

            "S1_final_url": None,
            "S2_final_url": None,
            "S3_final_url": None,

            "S1_temp_c": None,
            "S1_temp_f": None,
            "S1_method": None,
            "S1_error": None,
            "S1_match_text": None,

            "S2_temp_c": None,
            "S2_temp_f": None,
            "S2_method": None,
            "S2_error": None,
            "S2_match_text": None,

            "S3_temp_c": None,
            "S3_temp_f": None,
            "S3_method": None,
            "S3_error": None,
            "S3_match_text": None,
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

                results[code][f"{source}_final_url"] = row["final_url"]
                results[code][f"{source}_error"] = row["error"]
                results[code][f"{source}_temp_c"] = row["temp_c"]
                results[code][f"{source}_temp_f"] = row["temp_f"]
                results[code][f"{source}_method"] = row["method"]
                results[code][f"{source}_match_text"] = row["match_text"]

    return list(results.values())


def main():
    rows = process_all()

    for row in rows:
        print(f'\n=== {row["code"]} | {row["airport"]} ===')
        print("Local now:", row["local_now"])

        print("S1 URL      :", row["S1_url"])
        print("S1 FINAL URL:", row["S1_final_url"])
        print("S1 RESULT   :", row["S1_temp_c"], "C |", row["S1_temp_f"], "F |", row["S1_method"], "|", row["S1_error"])
        print("S1 MATCH    :", row["S1_match_text"])

        print("S2 URL      :", row["S2_url"])
        print("S2 FINAL URL:", row["S2_final_url"])
        print("S2 RESULT   :", row["S2_temp_c"], "C |", row["S2_temp_f"], "F |", row["S2_method"], "|", row["S2_error"])
        print("S2 MATCH    :", row["S2_match_text"])

        print("S3 URL      :", row["S3_url"])
        print("S3 FINAL URL:", row["S3_final_url"])
        print("S3 RESULT   :", row["S3_temp_c"], "C |", row["S3_temp_f"], "F |", row["S3_method"], "|", row["S3_error"])
        print("S3 MATCH    :", row["S3_match_text"])

    print("\nFINAL_JSON =")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
