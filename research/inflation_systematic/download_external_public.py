#!/usr/bin/env python3
"""Download and condense public data for real-time U.S. inflation forecasting.

The script runs in GitHub Actions, where outbound internet access is available.
It produces compact, provenance-stamped features rather than committing large
raw archives. Primary forecast origins are Philadelphia Fed SPF survey
quarters. To avoid using information released after the survey, FRED-MD uses
the monthly vintage immediately preceding the usual SPF survey month
(January/April/July/October for Q1/Q2/Q3/Q4 surveys).
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "inflation_systematic" / "external_data"
RAW = OUT / "raw"
PROCESSED = OUT / "processed"
for p in (RAW, PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; academic inflation forecasting replication; contact: repository owner)",
    "Accept": "*/*",
})
MANIFEST: list[dict[str, object]] = []


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, path: Path, *, optional: bool = False, timeout: int = 180) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = SESSION.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        data = response.content
        if len(data) < 20:
            raise RuntimeError(f"response too small: {len(data)} bytes")
        path.write_bytes(data)
        MANIFEST.append({
            "url": url,
            "final_url": response.url,
            "path": str(path.relative_to(ROOT)),
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "content_type": response.headers.get("content-type"),
            "status": "downloaded",
        })
        print(f"downloaded {url} -> {path} ({len(data):,} bytes)", flush=True)
        return path
    except Exception as exc:
        MANIFEST.append({"url": url, "path": str(path.relative_to(ROOT)), "status": "failed", "error": repr(exc)})
        if optional:
            print(f"optional download failed: {url}: {exc}", file=sys.stderr, flush=True)
            return None
        raise


def save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.name.endswith(".csv.gz"):
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)
    MANIFEST.append({
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "derived",
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    })


FRED_MD_ARCHIVES = [
    ("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical_fred-md.zip", RAW / "fred_md_1999_2014.zip"),
    ("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2025-12.zip", RAW / "fred_md_2015_2025.zip"),
]
FRED_QD_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-qd-2018-05-to-2025-12.zip"


def parse_fred_md_csv(data: bytes) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(io.BytesIO(data))
    date_col = raw.columns[0]
    tcodes = pd.to_numeric(raw.iloc[0, 1:], errors="coerce")
    tcodes.index = raw.columns[1:]
    df = raw.iloc[1:].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df, tcodes


def transform_series(series: pd.Series, code: int) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if code == 1:
        return x
    if code == 2:
        return x.diff()
    if code == 3:
        return x.diff().diff()
    if code == 4:
        return np.log(x.where(x > 0))
    if code == 5:
        return np.log(x.where(x > 0)).diff()
    if code == 6:
        return np.log(x.where(x > 0)).diff().diff()
    if code == 7:
        return x.pct_change(fill_method=None)
    return x.pct_change(fill_method=None)


def survey_quarters(start: str = "2000Q1", end: str = "2025Q4") -> list[pd.Period]:
    return list(pd.period_range(start, end, freq="Q"))


def conservative_vintage_month(quarter: pd.Period) -> str:
    month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter.quarter]
    return f"{quarter.year}-{month:02d}"


def collect_zip_csvs(paths: Iterable[Path]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                match = re.fullmatch(r"(\d{4}-\d{2})\.csv", Path(name).name, flags=re.I)
                if match:
                    found[match.group(1)] = archive.read(name)
    return found


def build_fred_md_features() -> None:
    archives = [download(url, path) for url, path in FRED_MD_ARCHIVES]
    files = collect_zip_csvs([path for path in archives if path is not None])
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for quarter in survey_quarters():
        vintage = conservative_vintage_month(quarter)
        data = files.get(vintage)
        if data is None:
            missing.append(vintage)
            continue
        frame, tcodes = parse_fred_md_csv(data)
        transformed = pd.DataFrame(index=frame.index)
        for column in frame.columns:
            tcode = tcodes.get(column, np.nan)
            code = int(tcode) if pd.notna(tcode) else 5
            transformed[column] = transform_series(frame[column], code)
        record: dict[str, object] = {"survey_q": str(quarter), "fred_md_vintage": vintage}
        for column in transformed.columns:
            values = transformed[column].dropna()
            if values.empty:
                continue
            record[f"fredmd__{column}__last"] = float(values.iloc[-1])
            record[f"fredmd__{column}__mean3"] = float(values.tail(3).mean())
            record[f"fredmd__{column}__mean6"] = float(values.tail(6).mean())
        rows.append(record)
    save_frame(pd.DataFrame(rows).sort_values("survey_q"), PROCESSED / "fred_md_survey_vintage_features.parquet")
    (PROCESSED / "fred_md_missing_vintages.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")

    qd_path = download(FRED_QD_URL, RAW / "fred_qd_2018_2025.zip", optional=True)
    if qd_path:
        with zipfile.ZipFile(qd_path) as archive:
            inventory = [{"name": item.filename, "bytes": item.file_size} for item in archive.infolist()]
        (PROCESSED / "fred_qd_archive_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")


FED_BOARD_FILES = {
    "cie": "https://www.federalreserve.gov/econres/notes/feds-notes/FEDS-Note-2873-cie-data.csv",
    "dkw": "https://www.federalreserve.gov/econres/notes/feds-notes/DKW_updates.csv",
    "ebp": "https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/files/ebp_csv.csv",
    "fcig": "https://www.federalreserve.gov/econres/notes/feds-notes/fci_g_public_monthly_3yr.csv",
    "scb_sentiment": "https://www.federalreserve.gov/econres/notes/feds-notes/SCB_Sentiment_Figure_2.csv",
    "tips_curve": "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv",
}
FRED_SERIES = [
    "T5YIE", "T10YIE", "T5YIFR", "DCOILWTICO", "DCOILBRENTEU", "GASREGW",
    "DHHNGSP", "DTWEXBGS", "VIXCLS", "SP500", "BAMLH0A0HYM2", "BAA10Y",
    "DGS2", "DGS5", "DGS10", "DFII5", "DFII10", "FEDFUNDS",
]


def build_market_files() -> None:
    for key, url in FED_BOARD_FILES.items():
        download(url, RAW / "fed_board" / f"{key}.csv", optional=True)
    for series_id in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        download(url, RAW / "fred_series" / f"{series_id}.csv", optional=True)
    download(
        "https://www.clevelandfed.org/-/media/files/webcharts/inflationexpectations/inflation-expectations.xlsx",
        RAW / "cleveland" / "inflation_expectations.xlsx",
        optional=True,
    )

    frames: list[pd.DataFrame] = []
    for path in sorted((RAW / "fred_series").glob("*.csv")):
        try:
            frame = pd.read_csv(path)
            if frame.shape[1] < 2:
                continue
            date_col, value_col = frame.columns[:2]
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
            frame[value_col] = pd.to_numeric(frame[value_col].replace(".", np.nan), errors="coerce")
            series = frame.dropna(subset=[date_col]).set_index(date_col)[value_col].rename(path.stem)
            frames.append(series.to_frame())
        except Exception as exc:
            print(f"could not parse {path}: {exc}", file=sys.stderr)
    if not frames:
        return
    daily = pd.concat(frames, axis=1).sort_index()
    records: list[dict[str, object]] = []
    for quarter in survey_quarters():
        month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter.quarter]
        cutoff = pd.Timestamp(quarter.year, month, 1) + pd.offsets.MonthEnd(0)
        history = daily.loc[:cutoff]
        record: dict[str, object] = {"survey_q": str(quarter), "cutoff_date": cutoff.date().isoformat()}
        for column in daily.columns:
            values = history[column].dropna()
            if values.empty:
                continue
            record[f"market__{column}__last"] = float(values.iloc[-1])
            record[f"market__{column}__mean21"] = float(values.tail(21).mean())
            if len(values) >= 22:
                record[f"market__{column}__chg21"] = float(values.iloc[-1] - values.iloc[-22])
            if len(values) >= 66:
                record[f"market__{column}__chg63"] = float(values.iloc[-1] - values.iloc[-64])
        records.append(record)
    save_frame(pd.DataFrame(records), PROCESSED / "market_survey_date_features.parquet")


BLS_IO_URL = "https://www.bls.gov/emp/input-output/input-output.zip"


def build_io_inventory() -> None:
    path = download(BLS_IO_URL, RAW / "bls_io" / "input-output.zip", optional=True)
    if path is None:
        return
    with zipfile.ZipFile(path) as archive:
        inventory = [{"name": item.filename, "bytes": item.file_size} for item in archive.infolist()]
        (PROCESSED / "bls_io_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        selected = []
        for item in archive.infolist():
            lower = item.filename.lower()
            if item.is_dir():
                continue
            if any(key in lower for key in ["sector", "layout", "description", "2024", "2023"]):
                if item.file_size <= 20_000_000:
                    selected.append(item)
        extract_root = RAW / "bls_io" / "selected"
        extract_root.mkdir(parents=True, exist_ok=True)
        for item in selected:
            (extract_root / Path(item.filename).name).write_bytes(archive.read(item.filename))


ZILLOW_CANDIDATES = [
    "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv",
    "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv",
    "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondo_sm_sa_month.csv",
]
APARTMENT_LIST_PAGE = "https://www.apartmentlist.com/research/category/data-rent-estimates"
ATLANTA_BIE_PAGE = "https://www.atlantafed.org/research-and-data/surveys/business-inflation-expectations"


def scrape_download_links(page_url: str, patterns: tuple[str, ...]) -> list[str]:
    try:
        response = SESSION.get(page_url, timeout=120)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = requests.compat.urljoin(response.url, anchor["href"])
            if any(pattern.lower() in href.lower() for pattern in patterns):
                links.append(href)
        return sorted(set(links))
    except Exception as exc:
        print(f"link scrape failed {page_url}: {exc}", file=sys.stderr)
        return []


def build_rent_business_files() -> None:
    for index, url in enumerate(ZILLOW_CANDIDATES):
        path = download(url, RAW / "zillow" / f"zori_candidate_{index}.csv", optional=True)
        if path is not None:
            try:
                frame = pd.read_csv(path)
                if len(frame) > 0 and frame.shape[1] > 20:
                    shutil.copy2(path, RAW / "zillow" / "zori_metro.csv")
                    break
            except Exception:
                pass
    apartment_links = scrape_download_links(APARTMENT_LIST_PAGE, (".csv", ".zip", "download"))
    (PROCESSED / "apartment_list_links.json").write_text(json.dumps(apartment_links, indent=2), encoding="utf-8")
    for index, url in enumerate(apartment_links[:10]):
        suffix = Path(url.split("?")[0]).suffix or ".dat"
        download(url, RAW / "apartment_list" / f"file_{index}{suffix}", optional=True)
    business_links = scrape_download_links(ATLANTA_BIE_PAGE, (".csv", ".xlsx", ".xls"))
    (PROCESSED / "atlanta_bie_links.json").write_text(json.dumps(business_links, indent=2), encoding="utf-8")
    for index, url in enumerate(business_links[:20]):
        suffix = Path(url.split("?")[0]).suffix or ".dat"
        download(url, RAW / "atlanta_bie" / f"file_{index}{suffix}", optional=True)


BEIGE_ARCHIVE = "https://www.federalreserve.gov/monetarypolicy/beige-book-archive.htm"
PRICE_TERMS = {
    "price_up": re.compile(r"\b(price increases?|raising prices?|higher prices?|pricing power|pass(?:ed|ing)? through)\b", re.I),
    "cost_up": re.compile(r"\b(input costs?|cost pressures?|higher costs?|wage pressures?|labor costs?)\b", re.I),
    "supply_tight": re.compile(r"\b(shortages?|supply chain|bottlenecks?|long lead times?|capacity constraints?)\b", re.I),
    "demand_up": re.compile(r"\b(strong demand|robust demand|demand increased|sales increased)\b", re.I),
    "price_down": re.compile(r"\b(price declines?|lower prices?|discounting|deflation|pricing pressure)\b", re.I),
    "supply_ease": re.compile(r"\b(supply chain improved|shortages eased|lead times shortened|normalizing supply)\b", re.I),
}


def build_beige_book_features() -> None:
    try:
        response = SESSION.get(BEIGE_ARCHIVE, timeout=120)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = requests.compat.urljoin(response.url, anchor["href"])
            label = (anchor.get_text(" ", strip=True) + " " + href).lower()
            if "beige" in label and any(extension in href.lower() for extension in [".htm", ".html", ".pdf"]):
                links.append(href)
        links = sorted(set(links))
        (PROCESSED / "beige_book_links.json").write_text(json.dumps(links, indent=2), encoding="utf-8")
        rows = []
        for url in links:
            if url.lower().endswith(".pdf"):
                continue
            try:
                page = SESSION.get(url, timeout=60)
                page.raise_for_status()
                page_soup = BeautifulSoup(page.text, "html.parser")
                text = page_soup.get_text(" ", strip=True)
                date_match = re.search(r"(19|20)\d{2}[-_/]?(0[1-9]|1[0-2])[-_/]?([0-3]\d)", url)
                date = None
                if date_match:
                    digits = re.sub(r"\D", "", date_match.group(0))
                    date = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
                words = max(1, len(re.findall(r"\b\w+\b", text)))
                record: dict[str, object] = {"url": url, "date": date, "words": words}
                for key, pattern in PRICE_TERMS.items():
                    record[key] = len(pattern.findall(text)) * 10000.0 / words
                rows.append(record)
            except Exception as exc:
                print(f"Beige Book parse failed {url}: {exc}", file=sys.stderr)
        if rows:
            save_frame(pd.DataFrame(rows), PROCESSED / "beige_book_price_features.csv")
    except Exception as exc:
        MANIFEST.append({"url": BEIGE_ARCHIVE, "status": "failed", "error": repr(exc)})
        print(f"Beige archive failed: {exc}", file=sys.stderr)


def main() -> None:
    build_fred_md_features()
    build_market_files()
    build_io_inventory()
    build_rent_business_files()
    build_beige_book_features()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "records": MANIFEST}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive_path = ROOT / "inflation_systematic_external_data.zip"
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in OUT.rglob("*"):
            if path.is_file():
                if path.name.endswith(".zip") and path.stat().st_size > 5_000_000:
                    continue
                archive.write(path, path.relative_to(ROOT))
    print(json.dumps({"artifact": str(archive_path), "bytes": archive_path.stat().st_size}, indent=2), flush=True)


if __name__ == "__main__":
    main()
