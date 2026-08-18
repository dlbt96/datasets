#!/usr/bin/env python3
"""Download Philadelphia Fed real-time CPI workbooks and export every sheet to CSV.

This helper is deliberately source-preserving: it records the requested URL,
HTTP metadata, SHA-256 digest, workbook sheet names, and an unmodified CSV
rendering of each sheet.  It is used to construct initial-release and vintage-
correct inflation targets for the PRICE survey-learning benchmark.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SOURCES = {
    "pcpi_first_second_third": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx/pcpi_first_second_third.xlsx",
    "pcpix_first_second_third": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx/pcpix_first_second_third.xlsx",
    "cpi_quarterly_vintages_monthly_observations": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx/cpiQvMd.xlsx",
}


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "sheet"


def main() -> None:
    out = Path("research/price_net/realtime_cpi")
    raw = out / "raw"
    csv_dir = out / "csv"
    raw.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 PRICE-NET academic replication"})
    records = []

    for key, url in SOURCES.items():
        response = session.get(url, timeout=120)
        response.raise_for_status()
        content = response.content
        xlsx_path = raw / f"{key}.xlsx"
        xlsx_path.write_bytes(content)
        sheets = pd.read_excel(xlsx_path, sheet_name=None, header=None, engine="openpyxl")
        exported = []
        for sheet_name, frame in sheets.items():
            path = csv_dir / f"{key}__{safe_name(str(sheet_name))}.csv"
            frame.to_csv(path, index=False, header=False)
            exported.append({"sheet": str(sheet_name), "path": str(path), "rows": int(len(frame)), "columns": int(frame.shape[1])})
        records.append({
            "key": key,
            "url": url,
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "sheets": exported,
        })

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Federal Reserve Bank of Philadelphia Real-Time Data Set",
        "records": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
