#!/usr/bin/env python3
"""Download public data for a real-time, data-rich U.S. inflation forecast.

The downloader is intentionally source-preserving. It stores the original bytes,
records URLs, timestamps, HTTP metadata, file sizes and SHA-256 hashes, and keeps
failures in the manifest rather than silently substituting another source.

Large public archives are downloaded as-is. Smaller web tables are normalized to
CSV in addition to preserving the source response. No proprietary data are used.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

OUT = Path("research/inflation_systematic/data")
RAW = OUT / "raw"
PROCESSED = OUT / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (compatible; UAlbany inflation forecasting research; "
            "+https://github.com/dlbt96/datasets)"
        )
    }
)

MANIFEST: list[dict[str, object]] = []


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "download"


def download(
    key: str,
    url: str,
    destination: Path,
    *,
    timeout: int = 240,
    optional: bool = False,
) -> Path | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "key": key,
        "url": url,
        "destination": str(destination),
        "optional": optional,
        "requested_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with SESSION.get(url, timeout=timeout, stream=True, allow_redirects=True) as response:
            response.raise_for_status()
            digest = hashlib.sha256()
            size = 0
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            record.update(
                {
                    "status": "ok",
                    "status_code": response.status_code,
                    "final_url": response.url,
                    "content_type": response.headers.get("content-type"),
                    "bytes": size,
                    "sha256": digest.hexdigest(),
                }
            )
            print(f"downloaded {key}: {size:,} bytes", flush=True)
            MANIFEST.append(record)
            return destination
    except Exception as exc:  # noqa: BLE001 - manifest must record all failures
        record.update({"status": "error", "error": repr(exc)})
        MANIFEST.append(record)
        print(f"FAILED {key}: {exc}", flush=True)
        if not optional:
            raise
        return None


def first_working_download(
    key: str,
    urls: Iterable[str],
    destination: Path,
    *,
    timeout: int = 240,
) -> Path | None:
    for index, url in enumerate(urls):
        result = download(
            f"{key}_attempt_{index + 1}",
            url,
            destination,
            timeout=timeout,
            optional=True,
        )
        if result is not None:
            return result
    return None


def normalize_html_tables(key: str, source_path: Path) -> None:
    try:
        tables = pd.read_html(source_path)
    except Exception as exc:  # noqa: BLE001
        MANIFEST.append(
            {
                "key": f"{key}_tables",
                "status": "error",
                "source": str(source_path),
                "error": repr(exc),
            }
        )
        return
    for idx, table in enumerate(tables):
        path = PROCESSED / f"{safe_name(key)}_table_{idx:02d}.csv"
        table.to_csv(path, index=False)
        MANIFEST.append(
            {
                "key": f"{key}_table_{idx:02d}",
                "status": "ok",
                "source": str(source_path),
                "destination": str(path),
                "rows": int(len(table)),
                "columns": int(table.shape[1]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )


def scrape_download_links(
    key: str,
    page_url: str,
    patterns: tuple[str, ...],
    destination_dir: Path,
    *,
    maximum: int = 30,
) -> list[Path]:
    page_path = destination_dir / f"{safe_name(key)}_page.html"
    if download(f"{key}_page", page_url, page_path, optional=True) is None:
        return []
    soup = BeautifulSoup(page_path.read_bytes(), "lxml")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(page_url, anchor["href"])
        text = " ".join(anchor.get_text(" ", strip=True).split())
        haystack = f"{href} {text}".lower()
        if any(re.search(pattern, haystack, flags=re.I) for pattern in patterns):
            links.append(href)
    # Stable de-duplication.
    unique: list[str] = []
    seen: set[str] = set()
    for link in links:
        if link not in seen:
            unique.append(link)
            seen.add(link)
    outputs: list[Path] = []
    for idx, link in enumerate(unique[:maximum]):
        name = Path(urlparse(link).path).name or f"link_{idx:02d}"
        destination = destination_dir / safe_name(name)
        result = download(f"{key}_link_{idx:02d}", link, destination, optional=True)
        if result is not None:
            outputs.append(result)
    return outputs


def download_fred_csv() -> None:
    series = [
        # Market expectations and risk prices.
        "T5YIE",
        "T10YIE",
        "T5YIFR",
        "T5YIEM",
        "T10YIEM",
        "EXPINF1YR",
        "EXPINF2YR",
        "EXPINF5YR",
        "EXPINF10YR",
        "DFII5",
        "DFII10",
        "DGS2",
        "DGS5",
        "DGS10",
        "BAMLH0A0HYM2",
        "VIXCLS",
        "DTWEXBGS",
        # High-frequency energy and commodities.
        "GASREGW",
        "GASALLW",
        "DCOILWTICO",
        "DCOILBRENTEU",
        "DHHNGSP",
        "WPU0561",
        "PPIACO",
        "PALLFNFINDEXQ",
        # Demand, supply, inventories and labor signals.
        "ICSA",
        "CCSA",
        "JTSJOL",
        "JTSQUR",
        "AWHAETP",
        "CES0500000003",
        "BUSINV",
        "ISRATIO",
        "RSXFS",
        "INDPRO",
        "TCU",
        "HOUST",
        "PERMIT",
    ]
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(series)
    download("fred_market_energy_activity", url, RAW / "fred" / "fred_market_energy_activity.csv")


def download_realtime_archives() -> None:
    sources = {
        "fred_md_vintages_1999_2014": (
            "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
            "fred-md/historical_fred-md.zip"
        ),
        "fred_md_vintages_2015_2025": (
            "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
            "fred-md/historical-vintages-of-fred-md-2015-01-to-2025-12.zip"
        ),
        "fred_qd_vintages_2018_2025": (
            "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
            "fred-md/historical-vintages-of-fred-qd-2018-05-to-2025-12.zip"
        ),
        "bls_input_output_1997_2024": (
            "https://www.bls.gov/emp/input-output/input-output.zip"
        ),
    }
    for key, url in sources.items():
        download(key, url, RAW / "archives" / f"{key}.zip", timeout=900, optional=True)


def download_expectations_sources() -> None:
    # Federal Reserve Board Common Inflation Expectations.
    scrape_download_links(
        "fed_cie",
        "https://www.federalreserve.gov/econres/notes/feds-notes/"
        "research-data-series-index-of-common-inflation-expectations-20210305.html",
        (r"cie.*\.csv", r"common inflation expectations.*csv"),
        RAW / "expectations" / "cie",
        maximum=8,
    )

    # Cleveland Fed expected-inflation term structure and firm survey.
    cleveland_page = RAW / "expectations" / "cleveland" / "inflation_expectations.html"
    if download(
        "cleveland_expected_inflation_page",
        "https://www.clevelandfed.org/indicators-and-data/inflation-expectations",
        cleveland_page,
        optional=True,
    ):
        scrape_download_links(
            "cleveland_expected_inflation",
            "https://www.clevelandfed.org/indicators-and-data/inflation-expectations",
            (r"\.xlsx", r"historical.*data", r"download.*data"),
            RAW / "expectations" / "cleveland",
            maximum=10,
        )

    sofie_page = RAW / "expectations" / "cleveland" / "sofie.html"
    if download(
        "cleveland_sofie_page",
        "https://www.clevelandfed.org/indicators-and-data/"
        "survey-of-firms-inflation-expectations",
        sofie_page,
        optional=True,
    ):
        normalize_html_tables("cleveland_sofie", sofie_page)

    # Atlanta Fed Business Inflation Expectations: preserve page and linked data.
    scrape_download_links(
        "atlanta_bie",
        "https://www.atlantafed.org/research/inflationproject/bie",
        (r"\.xlsx", r"\.csv", r"historical.*data", r"download"),
        RAW / "expectations" / "atlanta_bie",
        maximum=15,
    )


def download_rent_sources() -> None:
    first_working_download(
        "zillow_zori_national",
        (
            "https://files.zillowstatic.com/research/public_csvs/zori/"
            "Metro_zori_uc_sfrcondomfr_sm_sa_month.csv",
            "https://files.zillowstatic.com/research/public_csvs/zori/"
            "Metro_zori_uc_sfrcondomfr_sm_month.csv",
            "https://files.zillowstatic.com/research/public_csvs/zori/"
            "Metro_zori_sm_sa_month.csv",
        ),
        RAW / "rent" / "zillow_zori.csv",
    )
    apartment_page = RAW / "rent" / "apartment_list_page.html"
    if download(
        "apartment_list_rent_page",
        "https://www.apartmentlist.com/research/category/data-rent-estimates",
        apartment_page,
        optional=True,
    ):
        scrape_download_links(
            "apartment_list_rent",
            "https://www.apartmentlist.com/research/category/data-rent-estimates",
            (r"\.csv", r"rent estimates", r"download data"),
            RAW / "rent" / "apartment_list",
            maximum=15,
        )


def download_network_sources() -> None:
    # BLS IO is downloaded with the real-time archives. Scrape BEA's official IO
    # page for requirements/supply-use workbooks and preserve the linked files.
    scrape_download_links(
        "bea_input_output",
        "https://www.bea.gov/data/industries/input-output-accounts-data",
        (r"\.xlsx", r"requirements", r"supply.*use", r"use.*table"),
        RAW / "network" / "bea_io",
        maximum=25,
    )


def download_beige_book_index() -> None:
    # Full publication-time text is handled by a separate text workflow. This
    # step preserves the archive page and its document inventory.
    page_url = "https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm"
    page_path = RAW / "text" / "beige_book" / "archive_page.html"
    if download("beige_book_archive_page", page_url, page_path, optional=True) is None:
        return
    soup = BeautifulSoup(page_path.read_bytes(), "lxml")
    rows: list[dict[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(page_url, anchor["href"])
        text = " ".join(anchor.get_text(" ", strip=True).split())
        if "beige" in f"{href} {text}".lower() or re.search(r"\b(19|20)\d{2}\b", text):
            rows.append({"text": text, "url": href})
    pd.DataFrame(rows).drop_duplicates().to_csv(
        PROCESSED / "beige_book_link_inventory.csv", index=False
    )


def write_unavailable_data_registry() -> None:
    rows = [
        {
            "dataset": "NIQ/Circana scanner microprices",
            "access": "restricted/proprietary",
            "status": "not downloaded",
            "reason": "license or restricted-data agreement required",
        },
        {
            "dataset": "RealPage/CoStar/Yardi lease-level rents",
            "access": "proprietary",
            "status": "not downloaded",
            "reason": "commercial license required",
        },
        {
            "dataset": "Bloomberg/Blue Chip/Consensus Economics forecast vintages",
            "access": "proprietary",
            "status": "not downloaded",
            "reason": "commercial license required; official SPF is used instead",
        },
        {
            "dataset": "Inflation swaps, caps and floors",
            "access": "mostly proprietary",
            "status": "not downloaded",
            "reason": "reliable historical quote panels require a vendor",
        },
    ]
    pd.DataFrame(rows).to_csv(PROCESSED / "unavailable_proprietary_data.csv", index=False)


def main() -> None:
    download_realtime_archives()
    download_fred_csv()
    download_expectations_sources()
    download_rent_sources()
    download_network_sources()
    download_beige_book_index()
    write_unavailable_data_registry()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records": MANIFEST,
        "successful": sum(r.get("status") == "ok" for r in MANIFEST),
        "failed": sum(r.get("status") == "error" for r in MANIFEST),
    }
    path = OUT / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["created_utc", "successful", "failed"]}, indent=2))


if __name__ == "__main__":
    main()
