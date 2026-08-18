#!/usr/bin/env python3
"""Download and process Philadelphia Fed real-time CPI workbooks.

The pipeline preserves the source workbooks and worksheets, reconstructs
quarterly inflation from official first-release monthly growth, and constructs
vintage-correct targets from the quarterly-vintage CPI matrix. For a forecast
submitted in survey quarter s at horizon h, the maturity target is calculated
from vintage s+h—the first SPF survey quarter in which the complete endpoint
quarter is observable.
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


def parse_quarter_label(value: object) -> pd.Period | None:
    match = re.fullmatch(r"CPI(\d{2})Q([1-4])", str(value).strip(), flags=re.I)
    if not match:
        return None
    yy, quarter = int(match.group(1)), int(match.group(2))
    year = 1900 + yy if yy >= 65 else 2000 + yy
    return pd.Period(f"{year}Q{quarter}", freq="Q")


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
        z = z.loc[z[vintage_col] > -100].copy()
        z["log_growth"] = np.log1p(z[vintage_col] / 100.0) / 12.0
        z["index"] = 100.0 * np.exp(z["log_growth"].cumsum())
        z["quarter"] = z["date"].dt.to_period("Q")
        qlevel = z.groupby("quarter", sort=True)["index"].mean()
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


def maturity_vintage_targets(path: Path) -> pd.DataFrame:
    matrix = pd.read_excel(path, sheet_name="cpi", header=0, engine="openpyxl")
    date_col = matrix.columns[0]
    matrix[date_col] = pd.to_datetime(
        matrix[date_col].astype(str).str.replace(":", "-", regex=False) + "-01",
        errors="coerce",
    )
    matrix = matrix.dropna(subset=[date_col]).copy()
    matrix["observation_quarter"] = matrix[date_col].dt.to_period("Q")

    qlevels: dict[pd.Period, pd.Series] = {}
    for col in matrix.columns[1:]:
        vintage = parse_quarter_label(col)
        if vintage is None:
            continue
        values = pd.to_numeric(matrix[col], errors="coerce")
        levels = pd.DataFrame({"quarter": matrix["observation_quarter"], "level": values})
        qlevels[vintage] = levels.groupby("quarter", sort=True)["level"].mean()

    rows: list[dict[str, object]] = []
    survey_quarters = sorted(qlevels)
    for survey_q in survey_quarters:
        origin = survey_q - 1
        for h in [1, 2, 4]:
            maturity = survey_q + h
            if maturity not in qlevels:
                continue
            endpoint = origin + h
            levels = qlevels[maturity]
            base_level = levels.get(origin, np.nan)
            end_level = levels.get(endpoint, np.nan)
            if not (np.isfinite(base_level) and np.isfinite(end_level) and base_level > 0 and end_level > 0):
                continue
            target = (400.0 / h) * np.log(end_level / base_level)
            rows.append(
                {
                    "survey_q": str(survey_q),
                    "origin": str(origin),
                    "horizon_quarters": h,
                    "endpoint": str(endpoint),
                    "maturity_vintage": str(maturity),
                    "base_level": float(base_level),
                    "end_level": float(end_level),
                    "target_annualized_pct": float(target),
                }
            )
    return pd.DataFrame(rows).sort_values(["survey_q", "horizon_quarters"]).reset_index(drop=True)


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
            exported.append({"sheet": str(sheet_name), "path": str(path), "rows": int(len(frame)), "columns": int(frame.shape[1])})
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
    release_targets = pd.concat(
        [quarterly_targets(headline_monthly, "headline_cpi"), quarterly_targets(core_monthly, "core_cpi")],
        ignore_index=True,
    ).sort_values(["series", "vintage", "survey_q", "horizon_quarters"])
    release_path = processed_dir / "quarterly_inflation_targets.csv"
    release_targets.to_csv(release_path, index=False)

    maturity_targets = maturity_vintage_targets(raw_dir / "cpi_quarterly_vintages_monthly_observations.xlsx")
    maturity_path = processed_dir / "maturity_vintage_headline_targets.csv"
    maturity_targets.to_csv(maturity_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Federal Reserve Bank of Philadelphia Real-Time Data Set",
        "target_definition": "400/h times log quarterly-average CPI at origin+h over origin",
        "records": records,
        "release_target_path": str(release_path),
        "release_target_sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
        "maturity_target_path": str(maturity_path),
        "maturity_target_sha256": hashlib.sha256(maturity_path.read_bytes()).hexdigest(),
        "maturity_target_rows": int(len(maturity_targets)),
        "maturity_target_first_survey_q": maturity_targets["survey_q"].min(),
        "maturity_target_last_survey_q": maturity_targets["survey_q"].max(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
