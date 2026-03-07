import re
import json
import time
import html
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup


# =========================
# CONFIG
# =========================

AIRPORT_SEARCH = {
    "ATL": "Hartsfield Jackson Atlanta International Airport",
    "NYC": "LaGuardia Airport",
    "CHI": "O'Hare International Airport",
    "DAL": "Dallas Love Field",
    "SEA": "Seattle Tacoma International Airport",
    "MIA": "Miami International Airport",
    "TOR": "Toronto Pearson International Airport",
    "PAR": "Charles de Gaulle Airport",
    "SEL": "Incheon International Airport",
    "ANK": "Esenboga Airport",
    "BUE": "Ezeiza International Airport",
    "LON": "Heathrow Airport",
    "WLG": "Wellington Airport",
}

AIRPORT_TZ = {
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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

DDG_HTML = "https://html.duckduckgo.com/html/"


# =========================
# HELPERS
# =========================

def local_now(code: str) -> datetime:
    return datetime.now(ZoneInfo(AIRPORT_TZ[code]))


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def get(url: str, timeout: int = 25) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


def resolve_final_url(url: str) -> str:
    try:
        return get(url, timeout=25).url
    except Exception:
        return url


def ddg_search(query: str, pause: float = 1.0):
    time.sleep(pause)
    r = requests.post(
        DDG_HTML,
        headers=HEADERS,
        data={"q": query},
        timeout=30,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    results = []
    for a in soup.select("a.result__a"):
        href = a.get("href", "").strip()
        title = clean_text(a.get_text(" ", strip=True))
        if not href:
            continue

        # DuckDuckGo sometimes wraps the target URL inside uddg=
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            target = unquote(qs["uddg"][0])
        else:
            target = href

        results.append({
            "title": title,
            "url": target,
        })
    return results


def pick_best_url(candidates, rules):
    """
    rules = [
        lambda url,title: score,
        ...
    ]
    """
    scored = []
    for c in candidates:
        url = c["url"]
        title = c["title"]
        score = 0
        for fn in rules:
            try:
                score += fn(url, title)
            except Exception:
                pass
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]["url"] if scored and scored[0][0] > 0 else None


def maybe_force_accuweather_daily(url: str) -> str:
    if not url:
        return url
    if "accuweather.com" not in url:
        return url
    url = url.split("?")[0]
    if "/daily-weather-forecast/" in url:
        return url
    if "/weather-forecast/" in url:
        return url.replace("/weather-forecast/", "/daily-weather-forecast/")
    return url


def extract_json_scripts(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    blocks = []
    for s in soup.find_all("script"):
        txt = s.string or s.get_text()
        if txt:
            blocks.append(txt)
    return blocks


def first_int(s):
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else None


def f_to_c(f):
    return round((f - 32) * 5 / 9, 1)


# =========================
# LINK DISCOVERY
# =========================

def discover_accuweather_url(code: str, airport_name: str) -> str | None:
    queries = [
        f'site:accuweather.com "{airport_name}" "daily-weather-forecast"',
        f'site:accuweather.com "{airport_name}" "weather-forecast"',
        f'site:accuweather.com "{airport_name}" AccuWeather',
    ]

    candidates = []
    for q in queries:
        try:
            candidates.extend(ddg_search(q))
        except Exception:
            pass

    def rule(url, title):
        s = 0
        u = url.lower()
        t = title.lower()
        name = airport_name.lower()

        if "accuweather.com" in u:
            s += 100
        if "/daily-weather-forecast/" in u:
            s += 60
        if "/weather-forecast/" in u:
            s += 30
        if "airport" in u or "airport" in t:
            s += 15
        for token in re.split(r"[^a-z0-9]+", name):
            if token and token in u:
                s += 2
            if token and token in t:
                s += 2
        return s

    url = pick_best_url(candidates, [rule])
    if url:
        url = maybe_force_accuweather_daily(url)
        url = resolve_final_url(url)
        url = maybe_force_accuweather_daily(url)
    return url


def discover_weathercom_url(code: str, airport_name: str) -> str | None:
    queries = [
        f'site:weather.com/weather/tenday "{airport_name}"',
        f'site:weather.com "{airport_name}" "weather/tenday"',
        f'site:weather.com "{airport_name}" weather.com forecast',
    ]

    candidates = []
    for q in queries:
        try:
            candidates.extend(ddg_search(q))
        except Exception:
            pass

    def rule(url, title):
        s = 0
        u = url.lower()
        t = title.lower()
        name = airport_name.lower()

        if "weather.com" in u:
            s += 100
        if "/weather/tenday/" in u:
            s += 70
        if "/weather/today/" in u:
            s += 20
        if "airport" in u or "airport" in t:
            s += 10
        # hashed location style
        if re.search(r"/weather/tenday/l/[a-f0-9]{32,}", u):
            s += 30
        for token in re.split(r"[^a-z0-9]+", name):
            if token and token in u:
                s += 2
            if token and token in t:
                s += 2
        return s

    url = pick_best_url(candidates, [rule])
    if url:
        url = resolve_final_url(url)
    return url


def discover_bbc_url(code: str, airport_name: str) -> str | None:
    queries = [
        f'site:bbc.com/weather "{airport_name}"',
        f'site:bbc.com/weather "{airport_name}" BBC Weather',
        f'site:bbc.co.uk/weather "{airport_name}" BBC Weather',
    ]

    candidates = []
    for q in queries:
        try:
            candidates.extend(ddg_search(q))
        except Exception:
            pass

    def rule(url, title):
        s = 0
        u = url.lower()
        t = title.lower()
        name = airport_name.lower()

        if "bbc.com/weather" in u or "bbc.co.uk/weather" in u:
            s += 100
        if "forecast" in u:
            s += 20
        if "airport" in u or "airport" in t:
            s += 10
        for token in re.split(r"[^a-z0-9]+", name):
            if token and token in u:
                s += 2
            if token and token in t:
                s += 2
        return s

    url = pick_best_url(candidates, [rule])
    if url:
        url = resolve_final_url(url)
    return url


# =========================
# TEMP EXTRACTION
# =========================

def extract_accuweather_today_high(code: str, url: str):
    """
    Best-effort parser for AccuWeather daily page.
    Returns dict:
      {"temp_c": ..., "temp_f": ..., "method": "..."}
    """
    try:
        r = get(url, timeout=30)
        text = clean_text(BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True))

        now = local_now(code)
        weekday_short = now.strftime("%a")       # Sat
        m = now.month
        d = now.day

        # Pattern seen in accessible page text:
        # "Sat 3/7 27° /18° ..."
        patt1 = rf"\b{weekday_short}\s+{m}/{d}\s+(-?\d+)\s*[°º]\s*/\s*-?\d+\s*[°º]"
        m1 = re.search(patt1, text)
        if m1:
            c = int(m1.group(1))
            return {"temp_c": c, "temp_f": round(c * 9 / 5 + 32, 1), "method": "accuweather_day_row"}

        # Fallback: "Today 3/7 27° 18° ..."
        patt2 = rf"\bToday\s+{m}/{d}\s+(-?\d+)\s*[°º]"
        m2 = re.search(patt2, text, flags=re.I)
        if m2:
            c = int(m2.group(1))
            return {"temp_c": c, "temp_f": round(c * 9 / 5 + 32, 1), "method": "accuweather_today_label"}

        # Fallback generic - first hi/lo pair on page
        patt3 = r"\b(-?\d+)\s*[°º]\s*/\s*-?\d+\s*[°º]"
        m3 = re.search(patt3, text)
        if m3:
            c = int(m3.group(1))
            return {"temp_c": c, "temp_f": round(c * 9 / 5 + 32, 1), "method": "accuweather_first_pair"}

        return None
    except Exception:
        return None


def extract_weathercom_today_high(code: str, url: str):
    """
    Best-effort parser for Weather.com 10-day page.
    Weather.com often displays in F by default.
    """
    try:
        r = get(url, timeout=30)
        text = clean_text(BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True))
        now = local_now(code)
        weekday_short = now.strftime("%a")  # Sat
        day2 = now.strftime("%d")           # 07

        # Pattern seen in accessible text:
        # "Sat 07. 69° / 61° ..."
        patt1 = rf"\b{weekday_short}\s+{day2}\.\s+(-?\d+)\s*[°º]\s*/\s*-?\d+\s*[°º]"
        m1 = re.search(patt1, text)
        if m1:
            f = int(m1.group(1))
            return {"temp_c": f_to_c(f), "temp_f": f, "method": "weathercom_day_row"}

        # Sometimes "Today. -- / 59°" happens at night, so look for first proper day pair after current row
        patt2 = r"\b(?:Today|Tonight)\.?\s+--\s*/\s*-?\d+\s*[°º].{0,200}?\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{2}\.\s+(-?\d+)\s*[°º]\s*/\s*-?\d+\s*[°º]"
        m2 = re.search(patt2, text, flags=re.I)
        if m2:
            f = int(m2.group(1))
            return {"temp_c": f_to_c(f), "temp_f": f, "method": "weathercom_after_tonight"}

        # Fallback first day pair
        patt3 = r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{2}\.\s+(-?\d+)\s*[°º]\s*/\s*-?\d+\s*[°º]"
        m3 = re.search(patt3, text)
        if m3:
            f = int(m3.group(1))
            return {"temp_c": f_to_c(f), "temp_f": f, "method": "weathercom_first_pair"}

        return None
    except Exception:
        return None


def extract_bbc_today_high(code: str, url: str):
    """
    BBC parser is more heuristic because page structure can vary.
    We try:
    1) embedded JSON/script text
    2) raw visible text around today's local weekday/date
    """
    try:
        r = get(url, timeout=30)
        html_text = r.text
        text = clean_text(BeautifulSoup(html_text, "html.parser").get_text(" ", strip=True))
        now = local_now(code)
        weekday_full = now.strftime("%A")
        day_num = str(now.day)

        # Try embedded JSON snippets for max temp values
        scripts = extract_json_scripts(html_text)
        for block in scripts:
            # general patterns that often appear in weather JSON
            for patt in [
                r'"maxTempC"\s*:\s*(-?\d+)',
                r'"maximumTemperature"\s*:\s*(-?\d+)',
                r'"temperatureMax"\s*:\s*(-?\d+)',
            ]:
                m = re.search(patt, block)
                if m:
                    c = int(m.group(1))
                    return {"temp_c": c, "temp_f": round(c * 9 / 5 + 32, 1), "method": "bbc_script_json"}

        # Visible text fallback
        patt1 = rf"\b{weekday_full}\b.*?\b{day_num}\b.*?\bHigh\b.*?(-?\d+)\s*[°º]C"
        m1 = re.search(patt1, text, flags=re.I)
        if m1:
            c = int(m1.group(1))
            return {"temp_c": c, "temp_f": round(c * 9 / 5 + 32, 1), "method": "bbc_visible_high_c"}

        patt2 = r"\bHigh\b.*?(-?\d+)\s*[°º]C"
        m2 = re.search(patt2, text, flags=re.I)
        if m2:
            c = int(m2.group(1))
            return {"temp_c": c, "temp_f": round(c * 9 / 5 + 32, 1), "method": "bbc_first_high_c"}

        return None
    except Exception:
        return None


# =========================
# MAIN PIPELINE
# =========================

def process_airport(code: str, airport_name: str):
    out = {
        "code": code,
        "airport": airport_name,
        "local_now": local_now(code).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "S1_accuweather_url": None,
        "S2_bbc_url": None,
        "S3_weathercom_url": None,
        "S1_accuweather_today_high_c": None,
        "S1_accuweather_today_high_f": None,
        "S1_accuweather_method": None,
        "S2_bbc_today_high_c": None,
        "S2_bbc_today_high_f": None,
        "S2_bbc_method": None,
        "S3_weathercom_today_high_c": None,
        "S3_weathercom_today_high_f": None,
        "S3_weathercom_method": None,
        "status": "ok",
    }

    try:
        # S1 AccuWeather
        s1 = discover_accuweather_url(code, airport_name)
        out["S1_accuweather_url"] = s1
        if s1:
            temp = extract_accuweather_today_high(code, s1)
            if temp:
                out["S1_accuweather_today_high_c"] = temp["temp_c"]
                out["S1_accuweather_today_high_f"] = temp["temp_f"]
                out["S1_accuweather_method"] = temp["method"]

        # S2 BBC
        s2 = discover_bbc_url(code, airport_name)
        out["S2_bbc_url"] = s2
        if s2:
            temp = extract_bbc_today_high(code, s2)
            if temp:
                out["S2_bbc_today_high_c"] = temp["temp_c"]
                out["S2_bbc_today_high_f"] = temp["temp_f"]
                out["S2_bbc_method"] = temp["method"]

        # S3 Weather.com
        s3 = discover_weathercom_url(code, airport_name)
        out["S3_weathercom_url"] = s3
        if s3:
            temp = extract_weathercom_today_high(code, s3)
            if temp:
                out["S3_weathercom_today_high_c"] = temp["temp_c"]
                out["S3_weathercom_today_high_f"] = temp["temp_f"]
                out["S3_weathercom_method"] = temp["method"]

    except Exception as e:
        out["status"] = f"error: {type(e).__name__}: {e}"

    return out


def main():
    rows = []
    for code, airport_name in AIRPORT_SEARCH.items():
        print(f"\n=== {code} | {airport_name} ===")
        row = process_airport(code, airport_name)
        rows.append(row)

        print("S1 AccuWeather:", row["S1_accuweather_url"])
        print("S2 BBC        :", row["S2_bbc_url"])
        print("S3 Weather.com:", row["S3_weathercom_url"])
        print(
            "Today highs   :",
            {
                "S1_C": row["S1_accuweather_today_high_c"],
                "S2_C": row["S2_bbc_today_high_c"],
                "S3_C": row["S3_weathercom_today_high_c"],
            }
        )

    # pretty json
    print("\n\nFINAL_JSON =")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
