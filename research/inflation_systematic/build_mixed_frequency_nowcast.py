#!/usr/bin/env python3
"""Build exact-SPF-deadline mixed-frequency inflation nowcasts.

Public sources:
- Philadelphia Fed SPF deadline history, official mean/median CPI paths,
  and first-release headline/core CPI monthly growth rates.
- EIA weekly U.S. regular gasoline prices and daily Brent spot prices.

The output is a source-preserving panel of high-frequency energy features and a
transparent Cleveland-Fed-inspired monthly bridge model.  The model is not a
claim of an exact Cleveland Fed replication; its purpose is to create an
auditable public benchmark at the identical information date as the SPF.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research" / "inflation_systematic" / "mixed_frequency"
RAW = BASE / "raw"
OUT = BASE / "processed"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "spf_dates": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/spf-release-dates.txt",
    "spf_mean": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/historical-data/meanLevel.xlsx",
    "spf_median": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/historical-data/medianLevel.xlsx",
    "pcpi_first": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx/pcpi_first_second_third.xlsx",
    "pcpix_first": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx/pcpix_first_second_third.xlsx",
    "eia_gas_weekly": "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGw.xls",
    "eia_brent_daily": "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls",
}
HORIZONS = (1, 2, 4)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_all() -> list[dict[str, object]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 academic inflation forecasting replication"})
    records: list[dict[str, object]] = []
    suffix = {
        "spf_dates": ".txt", "spf_mean": ".xlsx", "spf_median": ".xlsx",
        "pcpi_first": ".xlsx", "pcpix_first": ".xlsx",
        "eia_gas_weekly": ".xls", "eia_brent_daily": ".xls",
    }
    for key, url in SOURCES.items():
        response = session.get(url, timeout=240)
        response.raise_for_status()
        data = response.content
        path = RAW / f"{key}{suffix[key]}"
        path.write_bytes(data)
        records.append({
            "key": key, "url": url, "bytes": len(data), "sha256": sha256(data),
            "content_type": response.headers.get("content-type"),
        })
        print("downloaded", key, len(data), flush=True)
    return records


def parse_deadlines(text: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_year: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.replace("\t", " ")
        match = re.match(
            r"\s*(?:(\d{4})\s+)?Q([1-4])\s+(\d{1,2}/\d{1,2}/\d{2})(?:\*+)?\s+(\d{1,2}/\d{1,2}/\d{2})(?:\*+)?",
            line,
        )
        if not match:
            continue
        if match.group(1):
            current_year = int(match.group(1))
        if current_year is None:
            continue
        quarter = int(match.group(2))
        deadline = pd.to_datetime(match.group(3), format="%m/%d/%y")
        release = pd.to_datetime(match.group(4), format="%m/%d/%y")
        rows.append({
            "survey_q": f"{current_year}Q{quarter}",
            "deadline": deadline,
            "release_date": release,
        })
    frame = pd.DataFrame(rows)
    frame = frame[(frame["deadline"].dt.year >= 1990)].drop_duplicates("survey_q")
    return frame.sort_values("deadline").reset_index(drop=True)


def parse_eia_history(path: Path, label: str) -> pd.Series:
    """Find the date column and the most populated numeric series in an EIA workbook."""
    sheets = pd.read_excel(path, sheet_name=None, header=None, engine="xlrd")
    best: tuple[int, pd.Series] | None = None
    for _, raw in sheets.items():
        if raw.empty:
            continue
        for date_col in raw.columns:
            dates = pd.to_datetime(raw[date_col], errors="coerce")
            n_dates = int(dates.notna().sum())
            if n_dates < 24:
                continue
            for value_col in raw.columns:
                if value_col == date_col:
                    continue
                values = pd.to_numeric(raw[value_col], errors="coerce")
                valid = dates.notna() & values.notna()
                score = int(valid.sum())
                if best is None or score > best[0]:
                    series = pd.Series(values.loc[valid].to_numpy(dtype=float), index=dates.loc[valid])
                    best = (score, series)
    if best is None:
        raise ValueError(f"Could not parse EIA workbook {path}")
    series = best[1]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = label
    return series


def parse_rtds_growth(path: Path) -> pd.Series:
    raw = pd.read_excel(path, sheet_name="DATA", header=None, engine="openpyxl")
    header_rows = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Date")]
    if len(header_rows) != 1:
        raise ValueError(f"Could not locate RTDS header in {path}")
    h = int(header_rows[0])
    frame = raw.iloc[h + 1 :, :5].copy()
    frame.columns = ["date", "first", "second", "third", "most_recent"]
    frame["date"] = pd.to_datetime(
        frame["date"].astype(str).str.replace(":", "-", regex=False) + "-01",
        errors="coerce",
    )
    frame["first"] = pd.to_numeric(frame["first"], errors="coerce")
    frame = frame.dropna(subset=["date", "first"]).sort_values("date")
    return frame.set_index("date")["first"].astype(float)


def load_spf_levels(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    candidates = []
    for _, frame in sheets.items():
        cols = {str(c).strip().upper(): c for c in frame.columns}
        if "YEAR" in cols and "QUARTER" in cols and "CPI2" in cols:
            z = frame.rename(columns={v: k for k, v in cols.items()}).copy()
            candidates.append(z)
    if not candidates:
        raise ValueError(f"No SPF CPI level sheet found in {path}")
    frame = max(candidates, key=len)
    frame["survey_q"] = [f"{int(y)}Q{int(q)}" for y, q in zip(frame["YEAR"], frame["QUARTER"])]
    keep = ["survey_q"] + [f"CPI{i}" for i in range(1, 7) if f"CPI{i}" in frame.columns]
    for c in keep[1:]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame[keep].drop_duplicates("survey_q").set_index("survey_q").sort_index()


def spf_paths(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for h in HORIZONS:
        cols = [f"CPI{i}" for i in range(2, 2 + h)]
        out[f"{prefix}_h{h}"] = frame[cols].mean(axis=1, skipna=False)
    for i in range(2, 6):
        if f"CPI{i}" in frame:
            out[f"{prefix}_q{i-1}"] = frame[f"CPI{i}"]
    return out


def reconstruct_index(annualized_monthly_rate: pd.Series, base: float = 100.0) -> pd.Series:
    rate = annualized_monthly_rate.sort_index().dropna()
    monthly_log = np.log1p(rate / 100.0) / 12.0
    return base * np.exp(monthly_log.cumsum())


def quarterly_rate_from_levels(levels: pd.Series, quarter: pd.Period) -> float:
    monthly = levels[levels.index.to_period("Q") == quarter]
    prev = levels[levels.index.to_period("Q") == quarter - 1]
    if len(monthly) != 3 or len(prev) != 3:
        return np.nan
    q, p = float(monthly.mean()), float(prev.mean())
    return 100.0 * ((q / p) ** 4.0 - 1.0)


def fit_linear(y: np.ndarray, X: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    penalty = np.eye(X1.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(X1.T @ X1 + penalty, X1.T @ y)


def gas_path_at_deadline(
    gas_weekly: pd.Series,
    brent_daily: pd.Series,
    deadline: pd.Timestamp,
    months: pd.DatetimeIndex,
) -> pd.Series:
    gas = gas_weekly.loc[:deadline]
    brent = brent_daily.loc[:deadline]
    if gas.empty or brent.empty:
        return pd.Series(index=months, dtype=float)
    gas_m = gas.resample("MS").mean()
    oil_m = brent.resample("MS").mean()
    complete_cutoff = deadline.to_period("M").start_time - pd.offsets.MonthBegin(1)
    hist = pd.concat([np.log(gas_m), np.log(oil_m)], axis=1, keys=["gas", "oil"]).loc[:complete_cutoff].dropna().tail(60)
    if len(hist) >= 24:
        beta = fit_linear(hist["gas"].to_numpy(), hist[["oil"]].to_numpy(), ridge=1e-4)
        resid = hist["gas"].to_numpy() - (beta[0] + beta[1] * hist["oil"].to_numpy())
        if len(resid) > 3 and np.var(resid[:-1]) > 1e-12:
            phi = float(np.clip(np.dot(resid[1:], resid[:-1]) / np.dot(resid[:-1], resid[:-1]), -0.95, 0.95))
        else:
            phi = 0.0
        last_resid = float(resid[-1])
    else:
        beta = np.array([float(np.log(gas.iloc[-1])), 0.0])
        phi, last_resid = 0.0, 0.0
    oil_last = float(np.log(brent.iloc[-1]))
    result = {}
    current_month = deadline.to_period("M")
    for month in months:
        period = month.to_period("M")
        observed = gas[(gas.index.to_period("M") == period)]
        if len(observed):
            result[month] = float(observed.mean())
        else:
            step = max(1, period.ordinal - current_month.ordinal + 1)
            predicted_resid = (phi ** step) * last_resid
            result[month] = float(np.exp(beta[0] + beta[1] * oil_last + predicted_resid))
    return pd.Series(result).sort_index()


def build_nowcasts(
    deadlines: pd.DataFrame,
    headline_rate: pd.Series,
    core_rate: pd.Series,
    gas_weekly: pd.Series,
    brent_daily: pd.Series,
    spf_mean: pd.DataFrame,
    spf_median: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    headline_level = reconstruct_index(headline_rate)
    rows = []
    features = []
    for rec in deadlines.itertuples(index=False):
        q = pd.Period(rec.survey_q, freq="Q")
        if q < pd.Period("2000Q1", freq="Q") or q > pd.Period("2026Q3", freq="Q"):
            continue
        deadline = pd.Timestamp(rec.deadline)
        history_end = q.start_time - pd.offsets.MonthBegin(1)
        h_head = headline_rate.loc[:history_end].dropna()
        h_core = core_rate.loc[:history_end].dropna()
        if len(h_head) < 72 or len(h_core) < 72:
            continue
        future_months = pd.date_range(q.start_time, (q + 3).end_time, freq="MS")
        gas_path = gas_path_at_deadline(gas_weekly, brent_daily, deadline, future_months)
        gas_hist = gas_weekly.loc[:deadline].resample("MS").mean()
        gas_all = pd.concat([gas_hist.loc[:history_end], gas_path]).sort_index()
        gas_growth = 1200.0 * np.log(gas_all / gas_all.shift(1))

        reg = pd.concat([h_head.rename("headline"), h_core.rename("core"), gas_growth.rename("gas")], axis=1).dropna().tail(84)
        if len(reg) < 36:
            continue
        beta = fit_linear(reg["headline"].to_numpy(), reg[["core", "gas"]].to_numpy(), ridge=1e-3)
        core_forecast = float(h_core.tail(12).mean())
        monthly_forecast = pd.Series(index=future_months, dtype=float)
        for month in future_months:
            g = float(gas_growth.get(month, np.nan))
            if not np.isfinite(g):
                g = 0.0
            monthly_forecast.loc[month] = float(beta[0] + beta[1] * core_forecast + beta[2] * g)

        projected_levels = headline_level.loc[:history_end].copy()
        last_level = float(projected_levels.iloc[-1])
        for month, rate in monthly_forecast.items():
            last_level *= math.exp(math.log1p(max(rate, -99.0) / 100.0) / 12.0)
            projected_levels.loc[month] = last_level
        q_rates = {j: quarterly_rate_from_levels(projected_levels, q + j - 1) for j in range(1, 5)}
        structural = {h: float(np.mean([q_rates[j] for j in range(1, h + 1)])) for h in HORIZONS}

        mean_row = spf_mean.loc[rec.survey_q] if rec.survey_q in spf_mean.index else None
        median_row = spf_median.loc[rec.survey_q] if rec.survey_q in spf_median.index else None
        row = {"survey_q": rec.survey_q, "deadline": deadline.date().isoformat()}
        for j in range(1, 5):
            row[f"mf_quarter_{j}"] = q_rates[j]
        for h in HORIZONS:
            row[f"mf_structural_h{h}"] = structural[h]
            if mean_row is not None and np.isfinite(mean_row.get("CPI2", np.nan)):
                future = [float(mean_row[f"CPI{i}"]) for i in range(3, 2 + h)] if h > 1 else []
                row[f"mf_spfmean_bridge_h{h}"] = float(np.mean([q_rates[1], *future]))
            if median_row is not None and np.isfinite(median_row.get("CPI2", np.nan)):
                future = [float(median_row[f"CPI{i}"]) for i in range(3, 2 + h)] if h > 1 else []
                row[f"mf_spfmedian_bridge_h{h}"] = float(np.mean([q_rates[1], *future]))
        rows.append(row)

        gobs = gas_weekly.loc[:deadline]
        oobs = brent_daily.loc[:deadline]
        f = {
            "survey_q": rec.survey_q,
            "deadline": deadline.date().isoformat(),
            "gas_last": float(gobs.iloc[-1]),
            "gas_mean4": float(gobs.tail(4).mean()),
            "gas_mean13": float(gobs.tail(13).mean()),
            "oil_last": float(oobs.iloc[-1]),
            "oil_mean21": float(oobs.tail(21).mean()),
            "core_roll12": core_forecast,
            "bridge_beta_core": float(beta[1]),
            "bridge_beta_gas": float(beta[2]),
        }
        if len(gobs) >= 14:
            f["gas_logchg13"] = float(100.0 * np.log(gobs.iloc[-1] / gobs.iloc[-14]))
        if len(oobs) >= 64:
            f["oil_logchg63"] = float(100.0 * np.log(oobs.iloc[-1] / oobs.iloc[-64]))
        features.append(f)
    nowcasts = pd.DataFrame(rows).sort_values("survey_q")
    feature_panel = pd.DataFrame(features).sort_values("survey_q")
    return nowcasts, feature_panel


def main() -> None:
    records = download_all()
    deadlines = parse_deadlines((RAW / "spf_dates.txt").read_text(encoding="utf-8", errors="ignore"))
    gas = parse_eia_history(RAW / "eia_gas_weekly.xls", "gas_weekly")
    brent = parse_eia_history(RAW / "eia_brent_daily.xls", "brent_daily")
    headline = parse_rtds_growth(RAW / "pcpi_first.xlsx")
    core = parse_rtds_growth(RAW / "pcpix_first.xlsx")
    mean = load_spf_levels(RAW / "spf_mean.xlsx")
    median = load_spf_levels(RAW / "spf_median.xlsx")
    mean_paths = spf_paths(mean, "spf_mean")
    median_paths = spf_paths(median, "spf_median")
    nowcasts, features = build_nowcasts(deadlines, headline, core, gas, brent, mean, median)

    deadlines.to_csv(OUT / "spf_deadlines.csv", index=False)
    gas.rename("gas_price").to_csv(OUT / "eia_weekly_gasoline.csv", index_label="date")
    brent.rename("brent_price").to_csv(OUT / "eia_daily_brent.csv", index_label="date")
    headline.rename("headline_first_release_annualized").to_csv(OUT / "headline_first_release_monthly.csv", index_label="date")
    core.rename("core_first_release_annualized").to_csv(OUT / "core_first_release_monthly.csv", index_label="date")
    pd.concat([mean_paths, median_paths], axis=1).reset_index(names="survey_q").to_csv(OUT / "official_spf_paths.csv", index=False)
    nowcasts.to_csv(OUT / "mixed_frequency_nowcasts.csv", index=False)
    features.to_csv(OUT / "mixed_frequency_features.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Exact-SPF-deadline public mixed-frequency energy nowcast and horizon-consistent SPF bridge",
        "method_boundary": "Cleveland-Fed-inspired public benchmark, not an exact official replication",
        "sources": records,
        "rows": {"nowcasts": int(len(nowcasts)), "features": int(len(features)), "deadlines": int(len(deadlines))},
        "first_survey_q": nowcasts["survey_q"].min(),
        "last_survey_q": nowcasts["survey_q"].max(),
    }
    (BASE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = ROOT / "inflation_systematic_mixed_frequency.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in BASE.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(ROOT))
    print(json.dumps({"artifact": str(archive), "bytes": archive.stat().st_size, **manifest}, indent=2))


if __name__ == "__main__":
    main()
