import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


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

# S1 = AccuWeather
# S2 = BBC
# S3 = Weather.com (uses lat/lon coords for reliable URL)
AIRPORT_LINKS = {
    "ATL": {
        "S1": "https://www.accuweather.com/en/us/hartsfield-jackson-atlanta-international-airport/30320/weather-forecast/9416_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/33.6407,-84.4277",
    },
    "NYC": {
        "S1": "https://www.accuweather.com/en/us/la-guardia-airport/11371/weather-forecast/26272_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/40.7772,-73.8726",
    },
    "CHI": {
        "S1": "https://www.accuweather.com/en/us/chicago-ohare-international-airport/60666/weather-forecast/72530_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/41.9742,-87.9073",
    },
    "DAL": {
        "S1": "https://www.accuweather.com/en/us/dallas-love-field/75235/weather-forecast/35119_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/32.8481,-96.8512",
    },
    "SEA": {
        "S1": "https://www.accuweather.com/en/us/seattle-tacoma-international-airport/98158/weather-forecast/351409_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/47.4502,-122.3088",
    },
    "MIA": {
        "S1": "https://www.accuweather.com/en/us/miami-international-airport/33122/weather-forecast/348308_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/25.7959,-80.2870",
    },
    "TOR": {
        "S1": "https://www.accuweather.com/en/ca/toronto-pearson-international-airport/l5p/weather-forecast/35185_poi",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/43.6777,-79.6248",
    },
    "PAR": {
        "S1": "https://www.accuweather.com/en/fr/roissy-en-france/135713/weather-forecast/135713",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/49.0097,2.5479",
    },
    "SEL": {
        "S1": "https://www.accuweather.com/en/kr/incheon/226081/weather-forecast/226081",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/37.4602,126.4407",
    },
    "ANK": {
        "S1": "https://www.accuweather.com/en/tr/esenboga-havalimani/318795/weather-forecast/318795",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/40.1281,32.9951",
    },
    "BUE": {
        "S1": "https://www.accuweather.com/en/ar/ezeiza/7894/weather-forecast/7894",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/-34.8222,-58.5358",
    },
    "LON": {
        "S1": "https://www.accuweather.com/en/gb/heathrow/328328/weather-forecast/328328",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/51.4775,-0.4614",
    },
    "WLG": {
        "S1": "https://www.accuweather.com/en/nz/wellington/250938/weather-forecast/250938",
        "S2": "",
        "S3": "https://weather.com/weather/tenday/l/-41.3272,174.8052",
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


def make_edge_driver() -> webdriver.Edge:
    """Create a headless Edge driver that mimics a real browser."""
    opts = webdriver.EdgeOptions()
    opts.add_argument("--headless")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Edge(options=opts)
    driver.execute_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    return driver


def fetch_weathercom_selenium(url: str):
    """
    Fetch weather.com page using Selenium/Edge and return
    (html_source, final_url, error_or_None).
    Each call creates and quits its own driver to stay thread-safe.
    """
    if not url:
        return None, None, "blank_url"
    driver = None
    try:
        driver = make_edge_driver()
        driver.get(url)
        # Wait until at least one TemperatureValue span is present
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="TemperatureValue"]'))
        )
        html = driver.page_source
        final_url = driver.current_url
        return html, final_url, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


# ============================================================
# S1 ACCUWEATHER
# ============================================================

def parse_accuweather_today_high(code: str, html: str):
    soup = BeautifulSoup(html, "html.parser")

    for idx, card in enumerate(soup.select("div.today-forecast-card")):
        for b in card.select("b"):
            txt = clean_text(b.get_text(" ", strip=True))
            m = re.search(r"\bHi\s*:\s*(-?\d+)", txt, flags=re.I)
            if m:
                c = int(m.group(1))
                return {"temp_c": c, "temp_f": c_to_f(c), "method": f"accuweather_today_card_b_{idx}", "match_text": txt}

        card_text = clean_text(card.get_text(" ", strip=True))
        m = re.search(r"\bHi\s*:\s*(-?\d+)", card_text, flags=re.I)
        if m:
            c = int(m.group(1))
            return {"temp_c": c, "temp_f": c_to_f(c), "method": f"accuweather_today_card_text_{idx}", "match_text": card_text[:300]}

    raw_patterns = [
        r'today-forecast-card.*?\bHi\s*:\s*(-?\d+)',
        r"\bTODAY'S WEATHER\b.{0,500}?\bHi\s*:\s*(-?\d+)",
        r"\bHi\s*:\s*(-?\d+)\s*°",
    ]
    for i, patt in enumerate(raw_patterns, start=1):
        m = re.search(patt, html, flags=re.I | re.S)
        if m:
            c = int(m.group(1))
            return {"temp_c": c, "temp_f": c_to_f(c), "method": f"accuweather_regex_{i}", "match_text": m.group(0)[:300]}

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
            return {"temp_c": c, "temp_f": c_to_f(c), "method": f"bbc_p{i}", "match_text": m.group(0)[:300]}

    return None


# ============================================================
# S3 WEATHER.COM
# NOTE: temperatures returned are in °F (weather.com US default).
#       We store the raw value in temp_f and convert to temp_c.
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
                    "temp_f": temp_val,
                    "temp_c": f_to_c(temp_val),
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
                    "temp_f": temp_val,
                    "temp_c": f_to_c(temp_val),
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
            f = int(m.group(1))
            return {
                "temp_f": f,
                "temp_c": f_to_c(f),
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
    # Weather.com (S3) requires Selenium; all others use requests
    if source == "S3":
        html, final_url, err = fetch_weathercom_selenium(url)
    else:
        html, final_url, err = fetch_text(session, url)

    if err:
        return {
            "code": code, "source": source, "url": url, "final_url": final_url,
            "ok": False, "error": err,
            "temp_c": None, "temp_f": None, "method": None, "match_text": None,
        }

    parsed = parse_source(code, source, html)
    if not parsed:
        return {
            "code": code, "source": source, "url": url, "final_url": final_url,
            "ok": False, "error": "parse_failed",
            "temp_c": None, "temp_f": None, "method": None, "match_text": None,
        }

    return {
        "code": code, "source": source, "url": url, "final_url": final_url,
        "ok": True, "error": None,
        "temp_c": parsed["temp_c"], "temp_f": parsed["temp_f"],
        "method": parsed["method"], "match_text": parsed.get("match_text"),
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
            "S1_final_url": None, "S2_final_url": None, "S3_final_url": None,
            "S1_temp_c": None, "S1_temp_f": None, "S1_method": None, "S1_error": None, "S1_match_text": None,
            "S2_temp_c": None, "S2_temp_f": None, "S2_method": None, "S2_error": None, "S2_match_text": None,
            "S3_temp_c": None, "S3_temp_f": None, "S3_method": None, "S3_error": None, "S3_match_text": None,
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
