"""
climate_project/scrape_epd.py

Scrapes daily AQI PDFs from EPD Punjab website.
URL: https://epd.punjab.gov.pk/aqi (links to PDFs per day)

Each PDF contains a table with AQI readings per monitoring station.
We extract Lahore station readings and convert AQI → PM2.5.

Coverage: Dec 1 2023 → Apr 15 2026 (gap in your existing dataset)
Output:   raw_data/epd_aqi_scraped.csv

Run: python scrape_epd.py

Notes:
  - The site is slow. Script sleeps 1s between requests to be polite.
  - PDFs that are missing on the server are silently skipped and logged.
  - Saves progress incrementally — safe to interrupt and resume.
  - pdfplumber handles scanned-style PDFs better than pypdf for tables.
"""

import os
import re
import time
import requests
import pdfplumber
import pandas as pd
from datetime import date, timedelta
from io import BytesIO
from urllib.parse import quote

os.makedirs("raw_data", exist_ok=True)
os.makedirs("raw_data/epd_pdfs", exist_ok=True)

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = date(2023, 12, 1)     # first date missing from your dataset
END_DATE   = date(2026, 4, 15)     # agreed cutoff

# ── URL patterns to try per date (in order) ───────────────────────────────────
# The site uses inconsistent naming — try all patterns before giving up
def url_patterns(d: date) -> list[str]:
    dd  = d.strftime("%d")
    mm  = d.strftime("%m")
    yy  = d.strftime("%Y")
    dmy = f"{dd}.{mm}.{yy}"
    ymd = f"{yy}.{mm}.{dd}"
    # URL-encode space as %20 for the space variant
    space_name = quote(f"EPA AQI {dmy}.pdf")
    return [
        f"https://epd.punjab.gov.pk/system/files/{dmy}.pdf",
        f"https://epd.punjab.gov.pk/system/files/EPA+AQI+{dmy}.pdf",
        f"https://epd.punjab.gov.pk/system/files/{space_name}",
        f"https://epd.punjab.gov.pk/system/files/EPA%20AQI%20{dmy}.pdf",
        f"https://epd.punjab.gov.pk/system/files/AQI+{dmy}.pdf",
        f"https://epd.punjab.gov.pk/system/files/AQI%20{dmy}.pdf",
        f"https://epd.punjab.gov.pk/system/files/{ymd}.pdf",
        f"https://epd.punjab.gov.pk/system/files?file={dmy}.pdf",
        f"https://epd.punjab.gov.pk/system/files?file=EPA+AQI+{dmy}.pdf",
    ]

# ── Lahore station keywords to match in the PDF ───────────────────────────────
LAHORE_KEYWORDS = [
    "lahore", "gulberg", "township", "iqbal town", "new garden town",
    "peco road", "us consulate", "jail road", "davis road",
    "egerton road", "shimla hill", "ferozepur road", "canal road",
    "mall road", "defence", "dha", "johar town",
]

# ── Column header patterns for station/location columns ──────────────────────
STATION_HEADERS = [
    "station", "location", "site", "city", "area",
    "aqms station", "aqms", "monitoring station", "district",
    "monitoring site", "name", "place",
]

# ── AQI breakpoints → PM2.5 (EPA formula, inverse) ───────────────────────────
AQI_BREAKPOINTS = [
    (0,   50,  0.0,   12.0),
    (51,  100, 12.1,  35.4),
    (101, 150, 35.5,  55.4),
    (151, 200, 55.5,  150.4),
    (201, 300, 150.5, 250.4),
    (301, 400, 250.5, 350.4),
    (401, 500, 350.5, 500.4),
]

def aqi_to_pm25(aqi: float) -> float | None:
    """Convert AQI value to PM2.5 µg/m³ using EPA linear interpolation."""
    if aqi is None or aqi < 0:
        return None
    for i_lo, i_hi, c_lo, c_hi in AQI_BREAKPOINTS:
        if i_lo <= aqi <= i_hi:
            return round(
                (c_hi - c_lo) / (i_hi - i_lo) * (aqi - i_lo) + c_lo, 1
            )
    # AQI > 500 — extrapolate linearly from last bracket
    if aqi > 500:
        return round(500.4 + (aqi - 500) * 0.99, 1)
    return None


def download_pdf(d: date, session: requests.Session) -> bytes | None:
    """Try each URL pattern, return raw PDF bytes or None."""
    for url in url_patterns(d):
        try:
            resp = session.get(url, timeout=15, allow_redirects=True)
            if resp.status_code == 200 and resp.headers.get(
                "Content-Type", ""
            ).startswith("application/pdf"):
                return resp.content
            # Some URLs return HTML 200 even for missing files
            if resp.status_code == 200 and b"%PDF" in resp.content[:10]:
                return resp.content
        except requests.RequestException:
            continue
    return None


def _find_col(headers: list[str], targets: list[str]) -> int | None:
    """Find the index of a column whose header matches any target keyword."""
    for i, h in enumerate(headers):
        h_clean = h.lower().strip().replace("\n", " ")
        for t in targets:
            if t in h_clean:
                return i
    return None


def extract_lahore_aqi(pdf_bytes: bytes, d: date) -> list[dict]:
    """
    Parse the PDF and extract AQI readings for Lahore stations.

    The EPD PDFs typically have a table with columns:
      Station | AQI | PM2.5 | PM10 | Category | ...

    Some older PDFs just list AQI without PM2.5 directly.
    We extract both where available, compute PM2.5 from AQI as fallback.

    Handles:
      - Multiple column header spellings
      - Merged cells (carry-forward station name from previous row)
      - Tables without explicit station column (join all cells, search text)
      - Scanned PDFs with looser text tolerance
    """
    rows = []
    try:
        # Use looser tolerances for scanned/noisy PDFs
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                # Try structured table extraction first
                # Try multiple extraction settings for robustness
                table_settings_variants = [
                    {},  # default
                    {"text_tolerance": 3, "snap_tolerance": 3},
                    {"text_tolerance": 5, "snap_tolerance": 5,
                     "join_tolerance": 5},
                ]
                tables_found = []
                for settings in table_settings_variants:
                    try:
                        tables = page.extract_tables(
                            table_settings=settings) if settings else page.extract_tables()
                        if tables and any(len(t) > 1 for t in tables if t):
                            tables_found = tables
                            break
                    except Exception:
                        continue

                last_station = ""  # for merged cell carry-forward

                for table in tables_found:
                    if not table or len(table) < 2:
                        continue
                    # Normalise headers
                    headers = [
                        str(h).lower().strip().replace("\n", " ")
                        if h else ""
                        for h in table[0]
                    ]

                    # Find column indices using expanded header list
                    station_col = _find_col(headers, STATION_HEADERS)
                    aqi_col = next(
                        (i for i, h in enumerate(headers)
                         if "aqi" in h and "pm" not in h), None
                    )
                    pm25_col = next(
                        (i for i, h in enumerate(headers)
                         if "pm2" in h or "pm 2" in h or "2.5" in h), None
                    )

                    for row in table[1:]:
                        if not row or all(c is None for c in row):
                            continue

                        # Determine station name
                        station = ""
                        if station_col is not None and station_col < len(row):
                            cell_val = str(row[station_col] or "").strip()
                            if cell_val:
                                station = cell_val.lower()
                                last_station = station  # update carry-forward
                            else:
                                # Merged cell — use previous row's station
                                station = last_station
                        else:
                            # No station column — join all cells and search
                            station = " ".join(
                                str(c or "") for c in row).lower()

                        # Check if this row is for a Lahore station
                        is_lahore = any(
                            kw in station for kw in LAHORE_KEYWORDS)
                        if not is_lahore:
                            continue

                        # Extract AQI
                        aqi_val = None
                        if aqi_col is not None and aqi_col < len(row):
                            raw = str(row[aqi_col] or "").strip()
                            nums = re.findall(r"\d+\.?\d*", raw)
                            if nums:
                                aqi_val = float(nums[0])

                        # Extract PM2.5 directly if available
                        pm25_val = None
                        if pm25_col is not None and pm25_col < len(row):
                            raw = str(row[pm25_col] or "").strip()
                            nums = re.findall(r"\d+\.?\d*", raw)
                            if nums:
                                pm25_val = float(nums[0])

                        # If no specific AQI col found, try to find a numeric
                        # value in any column that looks like an AQI
                        if aqi_val is None and pm25_val is None:
                            for ci, cell in enumerate(row):
                                if ci == station_col:
                                    continue
                                try:
                                    v = float(
                                        str(cell or "").strip().replace(
                                            ",", ""))
                                    if 10 <= v <= 600:
                                        aqi_val = v
                                        break
                                except (ValueError, TypeError):
                                    continue

                        # Derive PM2.5 from AQI if not directly available
                        if pm25_val is None and aqi_val is not None:
                            pm25_val = aqi_to_pm25(aqi_val)

                        if pm25_val is not None and 0 < pm25_val < 1500:
                            rows.append({
                                "date":    d.isoformat(),
                                "station": station.strip(),
                                "aqi":     aqi_val,
                                "pm25":    round(pm25_val, 1),
                                "source":  "EPD_PDF",
                            })

                # Fallback: raw text extraction if tables failed
                if not rows:
                    text = page.extract_text() or ""
                    text_lower = text.lower()

                    if any(kw in text_lower for kw in LAHORE_KEYWORDS):
                        # Look for AQI values near Lahore mentions
                        # Extended patterns for various PDF layouts
                        patterns = [
                            r"lahore[^0-9]*?(\d{2,4})",
                            r"(\d{2,4})\s*lahore",
                            r"gulberg[^0-9]*?(\d{2,4})",
                            r"township[^0-9]*?(\d{2,4})",
                            r"iqbal\s*town[^0-9]*?(\d{2,4})",
                            r"us\s*consulate[^0-9]*?(\d{2,4})",
                            r"jail\s*road[^0-9]*?(\d{2,4})",
                            r"peco\s*road[^0-9]*?(\d{2,4})",
                            r"new\s*garden[^0-9]*?(\d{2,4})",
                            # Catch "PM2.5: 123" or "AQI: 234" near Lahore
                            r"lahore.*?pm2\.?5[:\s]+(\d+\.?\d*)",
                            r"lahore.*?aqi[:\s]+(\d+\.?\d*)",
                        ]
                        for pat in patterns:
                            matches = re.findall(pat, text_lower)
                            for m in matches:
                                try:
                                    val = float(m)
                                    if 5 <= val <= 2000:
                                        # Guess: if > 100, likely AQI;
                                        # < 100 could be either
                                        pm25 = aqi_to_pm25(
                                            val) if val > 100 else val
                                        if pm25 and 0 < pm25 < 1500:
                                            rows.append({
                                                "date":    d.isoformat(),
                                                "station":
                                                    "lahore (text-extracted)",
                                                "aqi":
                                                    val if val > 100 else None,
                                                "pm25":    round(pm25, 1),
                                                "source":  "EPD_PDF_text",
                                            })
                                            break
                                except ValueError:
                                    continue
                            if rows:
                                break

    except Exception as e:
        print(f"    Parse error for {d}: {e}")

    # Post-filter: only keep rows that match Lahore keywords
    # (catches multi-city PDFs where station names are ambiguous)
    verified = [r for r in rows
                if any(kw in r["station"] for kw in LAHORE_KEYWORDS)]
    return verified if verified else rows


def daily_lahore_pm25(rows: list[dict], d: date) -> float | None:
    """Aggregate multiple station readings into one daily Lahore PM2.5."""
    vals = [r["pm25"] for r in rows if r["pm25"] is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def main():
    print("=" * 60)
    print("  EPD Punjab AQI Scraper")
    print(f"  {START_DATE} → {END_DATE}")
    print("=" * 60)

    # Load existing progress
    out_path = "raw_data/epd_aqi_scraped.csv"
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, parse_dates=["date"])
        done_dates = set(existing["date"].dt.date)
        results    = existing.to_dict("records")
        print(f"  Resuming from saved progress "
              f"({len(done_dates)} dates already done)")
    else:
        done_dates = set()
        results    = []

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (research project, Habib University)",
    })

    current = START_DATE
    skipped = []
    n_total = (END_DATE - START_DATE).days + 1

    while current <= END_DATE:
        if current in done_dates:
            current += timedelta(days=1)
            continue

        day_str = current.strftime("%d %b %Y")
        print(f"  [{current}] Fetching...", end=" ", flush=True)

        pdf_bytes = download_pdf(current, session)

        if pdf_bytes is None:
            print("no PDF found")
            skipped.append(current.isoformat())
            # Still record the date as attempted with None
            results.append({
                "date": current.isoformat(),
                "pm25": None,
                "aqi":  None,
                "n_stations": 0,
                "source": "EPD_PDF",
            })
        else:
            # Save raw PDF for reference
            pdf_path = f"raw_data/epd_pdfs/{current.isoformat()}.pdf"
            if not os.path.exists(pdf_path):
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)

            rows   = extract_lahore_aqi(pdf_bytes, current)
            pm25   = daily_lahore_pm25(rows, current)
            n_stat = len(rows)

            if pm25:
                print(f"PM2.5={pm25} µg/m³ ({n_stat} stations)")
            else:
                print(f"PDF found but no Lahore readings extracted")

            results.append({
                "date":       current.isoformat(),
                "pm25":       pm25,
                "aqi":        rows[0]["aqi"] if rows else None,
                "n_stations": n_stat,
                "source":     "EPD_PDF",
            })

        # Save progress every 10 days
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

        current += timedelta(days=1)
        time.sleep(1.0)   # polite rate limiting

    # Final save
    df = pd.DataFrame(results)
    df = df[df["pm25"].notna()]   # drop days with no data
    df.to_csv(out_path, index=False)

    print(f"\n  Done. {len(df)} days with PM2.5 data saved to {out_path}")
    print(f"  Skipped (no PDF): {len(skipped)} days")
    if skipped:
        print(f"  Missing dates: "
              f"{skipped[:10]}{'...' if len(skipped)>10 else ''}")

    print("\n  Coverage summary:")
    df["date"] = pd.to_datetime(df["date"])
    print(f"    Date range: {df.date.min().date()} → {df.date.max().date()}")
    print(f"    PM2.5 mean: {df.pm25.mean():.1f} µg/m³")
    print(f"    PM2.5 max:  {df.pm25.max():.1f} µg/m³")


if __name__ == "__main__":
    main()
