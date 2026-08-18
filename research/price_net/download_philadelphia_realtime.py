#!/usr/bin/env python3
"""Download and process Philadelphia Fed real-time CPI workbooks.

The source workbooks and every worksheet are preserved. Processed targets match
the SPF convention: each quarterly CPI forecast is a discretely compounded
quarter-over-quarter annualized percentage rate based on quarterly-average CPI,
and an h-quarter path is the arithmetic mean of the next h quarterly rates.

For a forecast submitted in survey quarter s at horizon h, the maturity target
uses vintage s+h, the first SPF survey quarter in which the complete endpoint
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
HORIZONS = (1, 2, 4)


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


def path_target_from_qlevels(qlevel: pd.Series, survey_q: pd.Period, horizon: int) -> float:
    """Average next-h SPF-style quarterly inflation rates."""
    rates: list[float] = []
    for step in range(horizon):
        quarter = survey_q + step
        previous = quarter - 1
        p0, p1 = qlevel.get(previous, np.nan), qlevel.get(quarter, np.nan)
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0):
            return np.nan
        rates.append(100.0 * ((p1 / p0) ** 4.0 - 1.0))
    return float(np.mean(rates))


def quarterly_targets(monthly: pd.DataFrame, series: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for vintage_col in ["first", "most_recent"]:
        z = monthly[["date", vintage_col]].dropna().copy()
        z = z.loc[z[vintage_col] > -100].copy()
        # Annualized discrete monthly rate -> monthly gross growth.
        monthly_gross = (1.0 + z[vintage_col] / 100.0) ** (1.0 / 12.0)
        z["index"] = 100.0 * monthly_gross.cumprod()
        z["quarter"] = z["date"].dt.to_period("Q")
        qlevel = z.groupby("quarter", sort=True)["index"].mean()
        for survey_q in qlevel.index:
            for horizon in HORIZONS:
                value = path_target_from_qlevels(qlevel, survey_q, horizon)
                if not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "series": series,
                        "vintage": "first_release" if vintage_col == "first" else "most_recent_release",
                        "origin": str(survey_q - 1),
                        "survey_q": str(survey_q),
                        "horizon_quarters": horizon,
                        "target_annualized_pct": value,
                    }
                )
    return pd.DataFrame(rows)


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
    for survey_q in sorted(qlevels):
        for horizon in HORIZONS:
            maturity = survey_q + horizon
            if maturity not in qlevels:
                continue
            value = path_target_from_qlevels(qlevels[maturity], survey_q, horizon)
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "survey_q": str(survey_q),
                    "origin": str(survey_q - 1),
                    "horizon_quarters": horizon,
                    "endpoint": str(survey_q + horizon - 1),
                    "maturity_vintage": str(maturity),
                    "target_annualized_pct": value,
                }
            )
    return pd.DataFrame(rows).sort_values(["survey_q", "horizon_quarters"]).reset_index(drop=True)


def main() -> None:
    out = Path("research/price_net/realtime_cpi")
    raw_dir, csv_dir, processed_dir = out / "raw", out / "csv", out / "processed"
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
        records.append({"key": key, "url": url, "status_code": response.status_code,
                        "content_type": response.headers.get("content-type"), "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(), "sheets": exported})

    release_targets = pd.concat(
        [
            quarterly_targets(parse_growth_sheet(raw_dir / "pcpi_first_second_third.xlsx"), "headline_cpi"),
            quarterly_targets(parse_growth_sheet(raw_dir / "pcpix_first_second_third.xlsx"), "core_cpi"),
        ],
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
        "target_definition": "Arithmetic mean of next-h quarterly CPI rates, each 100*((quarterly-average CPI_t/CPI_t-1)^4-1)",
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
