#!/usr/bin/env python3
"""Download and process Philadelphia Fed real-time CPI workbooks.

The pipeline preserves the source workbooks and every worksheet, then builds a
small quarterly target panel from the official first-release and most-recent
monthly CPI growth rates.  Monthly growth is reported at an annual rate, so a
chained index is reconstructed using a one-twelfth power before quarterly
averaging.  The resulting targets are aligned to SPF survey quarters.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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


def parse_growth_sheet(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="DATA", header=None, engine="openpyxl")
    header_rows = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Date")]
    if len(header_rows) != 1:
        raise ValueError(f"Could not uniquely locate Date header in {path}")
    h = int(header_rows[0])
    frame = raw.iloc[h + 1 :, :5].copy()
    frame.columns = ["date", "first", "second", "third", "most_recent"]
    frame["date"] = pd.to_datetime(
        frame["date"].astype(str).str.replace(":", "-", regex=False) + "-01",
        errors="coerce",
    )
    for col in ["first", "second", "third", "most_recent"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def quarterly_targets(monthly: pd.DataFrame, series: str) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for vintage_col in ["first", "most_recent"]:
        z = monthly[["date", vintage_col]].dropna().copy()
        # Annualized monthly percentage rate -> monthly log growth.
        valid = z[vintage_col] > -100
        z = z.loc[valid].copy()
        z["log_growth"] = np.log1p(z[vintage_col] / 100.0) / 12.0
        z["index"] = 100.0 * np.exp(z["log_growth"].cumsum())
        z["quarter"] = z["date"].dt.to_period("Q")
        qlevel = z.groupby("quarter", sort=True)["index"].mean()
        # A survey conducted in q forecasts inflation from q-1 through q-1+h.
        for h in [1, 2, 4]:
            target = (400.0 / h) * np.log(qlevel.shift(-h) / qlevel)
            for origin, value in target.dropna().items():
                output.append(
                    {
                        "series": series,
                        "vintage": "first_release" if vintage_col == "first" else "most_recent_release",
                        "origin": str(origin),
                        "survey_q": str(origin + 1),
                        "horizon_quarters": h,
                        "target_annualized_pct": float(value),
                    }
                )
    return pd.DataFrame(output)


def main() -> None:
    out = Path("research/price_net/realtime_cpi")
    raw_dir = out / "raw"
    csv_dir = out / "csv"
    processed_dir = out / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 PRICE-NET academic replication"})
    records = []

    for key, url in SOURCES.items():
        response = session.get(url, timeout=120)
        response.raise_for_status()
        content = response.content
        xlsx_path = raw_dir / f"{key}.xlsx"
        xlsx_path.write_bytes(content)
        sheets = pd.read_excel(xlsx_path, sheet_name=None, header=None, engine="openpyxl")
        exported = []
        for sheet_name, frame in sheets.items():
            path = csv_dir / f"{key}__{safe_name(str(sheet_name))}.csv"
            frame.to_csv(path, index=False, header=False)
            exported.append(
                {
                    "sheet": str(sheet_name),
                    "path": str(path),
                    "rows": int(len(frame)),
                    "columns": int(frame.shape[1]),
                }
            )
        records.append(
            {
                "key": key,
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "sheets": exported,
            }
        )

    headline_monthly = parse_growth_sheet(raw_dir / "pcpi_first_second_third.xlsx")
    core_monthly = parse_growth_sheet(raw_dir / "pcpix_first_second_third.xlsx")
    targets = pd.concat(
        [
            quarterly_targets(headline_monthly, "headline_cpi"),
            quarterly_targets(core_monthly, "core_cpi"),
        ],
        ignore_index=True,
    ).sort_values(["series", "vintage", "survey_q", "horizon_quarters"])
    target_path = processed_dir / "quarterly_inflation_targets.csv"
    targets.to_csv(target_path, index=False)

    target_summary = (
        targets.groupby(["series", "vintage", "horizon_quarters"])
        .agg(rows=("target_annualized_pct", "size"), first_survey_q=("survey_q", "min"), last_survey_q=("survey_q", "max"))
        .reset_index()
        .to_dict(orient="records")
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Federal Reserve Bank of Philadelphia Real-Time Data Set",
        "target_definition": "400/h times log quarterly-average chained index at origin+h over origin",
        "records": records,
        "processed_target_path": str(target_path),
        "processed_target_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "processed_target_summary": target_summary,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
