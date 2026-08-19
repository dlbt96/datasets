#!/usr/bin/env python3
"""Build the compact real-time macro/market feature block used by the forecast test."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research" / "inflation_systematic" / "core_external"
RAW = BASE / "raw"
OUT = BASE / "processed"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

ARCHIVES = [RAW / "fred_md_1999_2014.zip", RAW / "fred_md_2015_2025.zip"]
SURVEY_MONTH = {1: 1, 2: 4, 3: 7, 4: 10}
MARKET_SERIES = [
    "T5YIE", "T10YIE", "T5YIFR", "DCOILWTICO", "DCOILBRENTEU", "GASREGW",
    "DHHNGSP", "DTWEXBGS", "VIXCLS", "SP500", "BAMLH0A0HYM2", "BAA10Y",
    "DGS2", "DGS5", "DGS10", "DFII5", "DFII10", "FEDFUNDS",
]


def survey_quarters():
    return list(pd.period_range("2000Q1", "2025Q4", freq="Q"))


def vintage_for(q: pd.Period):
    return f"{q.year}-{SURVEY_MONTH[q.quarter]:02d}"


def read_zip_members():
    members = {}
    for path in ARCHIVES:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                base = Path(name).name
                if len(base) == 11 and base.endswith(".csv") and base[:7][4] == "-":
                    members[base[:7]] = (path, name)
    return members


def parse_vintage(data: bytes):
    raw = pd.read_csv(io.BytesIO(data), low_memory=False)
    date_col = raw.columns[0]
    tcodes = pd.to_numeric(raw.iloc[0, 1:], errors="coerce")
    tcodes.index = raw.columns[1:]
    df = raw.iloc[1:].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    transformed = pd.DataFrame(index=df.index)
    groups = {code: list(tcodes[tcodes == code].index.intersection(df.columns)) for code in range(1, 8)}
    if groups[1]: transformed[groups[1]] = df[groups[1]]
    if groups[2]: transformed[groups[2]] = df[groups[2]].diff()
    if groups[3]: transformed[groups[3]] = df[groups[3]].diff().diff()
    if groups[4]: transformed[groups[4]] = np.log(df[groups[4]].where(df[groups[4]] > 0))
    if groups[5]: transformed[groups[5]] = np.log(df[groups[5]].where(df[groups[5]] > 0)).diff()
    if groups[6]: transformed[groups[6]] = np.log(df[groups[6]].where(df[groups[6]] > 0)).diff().diff()
    if groups[7]: transformed[groups[7]] = df[groups[7]].pct_change(fill_method=None)
    unknown = [c for c in df.columns if c not in transformed.columns]
    if unknown: transformed[unknown] = df[unknown].pct_change(fill_method=None)
    return transformed


def build_fred_md():
    members = read_zip_members()
    rows = []
    missing = []
    for q in survey_quarters():
        vintage = vintage_for(q)
        if vintage not in members:
            missing.append(vintage)
            continue
        archive_path, member = members[vintage]
        with zipfile.ZipFile(archive_path) as zf:
            data = zf.read(member)
        frame = parse_vintage(data)
        record = {"survey_q": str(q), "fred_md_vintage": vintage}
        last = frame.ffill().iloc[-1]
        mean3 = frame.tail(3).mean()
        mean6 = frame.tail(6).mean()
        for c, value in last.items():
            if np.isfinite(value): record[f"fredmd__{c}__last"] = float(value)
        for c, value in mean3.items():
            if np.isfinite(value): record[f"fredmd__{c}__mean3"] = float(value)
        for c, value in mean6.items():
            if np.isfinite(value): record[f"fredmd__{c}__mean6"] = float(value)
        rows.append(record)
        print("processed", vintage, flush=True)
    df = pd.DataFrame(rows).sort_values("survey_q")
    path = OUT / "fred_md_survey_vintage_features.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    (OUT / "fred_md_missing.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
    return path


def build_market():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 academic inflation forecasting"})
    frames = []
    source_rows = []
    for series in MARKET_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            data = response.content
            (RAW / f"{series}.csv").write_bytes(data)
            source_rows.append({"series": series, "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "status": "downloaded"})
            frame = pd.read_csv(io.BytesIO(data))
            frame.iloc[:, 0] = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
            frame.iloc[:, 1] = pd.to_numeric(frame.iloc[:, 1].replace(".", np.nan), errors="coerce")
            frames.append(frame.dropna(subset=[frame.columns[0]]).set_index(frame.columns[0]).iloc[:, 0].rename(series))
        except Exception as exc:
            source_rows.append({"series": series, "url": url, "status": "failed", "error": repr(exc)})
    daily = pd.concat(frames, axis=1).sort_index()
    rows = []
    for q in survey_quarters():
        cutoff = pd.Timestamp(q.year, SURVEY_MONTH[q.quarter], 1) + pd.offsets.MonthEnd(0)
        hist = daily.loc[:cutoff]
        record = {"survey_q": str(q), "cutoff": cutoff.date().isoformat()}
        for c in daily.columns:
            x = hist[c].dropna()
            if x.empty: continue
            record[f"market__{c}__last"] = float(x.iloc[-1])
            record[f"market__{c}__mean21"] = float(x.tail(21).mean())
            if len(x) >= 22: record[f"market__{c}__chg21"] = float(x.iloc[-1] - x.iloc[-22])
            if len(x) >= 64: record[f"market__{c}__chg63"] = float(x.iloc[-1] - x.iloc[-64])
        rows.append(record)
    path = OUT / "market_survey_features.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    (OUT / "market_sources.json").write_text(json.dumps(source_rows, indent=2), encoding="utf-8")
    return path


def main():
    fred_path = build_fred_md()
    market_path = build_market()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fred_md": str(fred_path.relative_to(ROOT)),
        "market": str(market_path.relative_to(ROOT)),
        "timing": "January/April/July/October FRED-MD vintage and month-end market data before usual SPF survey month",
    }
    (BASE / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    archive = ROOT / "inflation_systematic_core_external.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in BASE.rglob("*"):
            if path.is_file() and path not in ARCHIVES:
                zf.write(path, path.relative_to(ROOT))
    print(json.dumps({"artifact": str(archive), "bytes": archive.stat().st_size, **metadata}, indent=2))

if __name__ == "__main__":
    main()
