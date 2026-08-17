#!/usr/bin/env python3
"""Build point-in-time monthly inflation signals from public earnings calls.

The script streams kurry/sp500_earnings_transcripts from Hugging Face, never
writes raw transcript text, and exports call-level economic indicators plus
monthly aggregates. The features are deliberately transparent and serve as the
first text baseline before supervised/LLM extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.feature_extraction.text import HashingVectorizer

DATASET_ID = "kurry/sp500_earnings_transcripts"

FORWARD_RE = re.compile(
    r"\b(expect(?:ed|ing)?|anticipat(?:e|ed|ing)|forecast|outlook|guidance|"
    r"plan(?:ned|ning)?|intend|will|going to|next (?:month|quarter|year)|"
    r"remainder of (?:the )?year|looking ahead|future)\b",
    re.I,
)

PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "final_price_up": tuple(re.compile(x, re.I) for x in [
        r"\brais(?:e|ed|ing) (?:our )?prices?\b", r"\bprice increases?\b",
        r"\bhigher (?:selling )?prices?\b", r"\bpricing actions?\b",
        r"\bpricing (?:power|benefit|contribution|realization)\b",
        r"\btak(?:e|ing|en) price\b", r"\bpass(?:ing)? (?:it )?through\b",
        r"\bsurcharge(?:s)?\b",
    ]),
    "final_price_down": tuple(re.compile(x, re.I) for x in [
        r"\blower (?:selling )?prices?\b", r"\bprice reductions?\b",
        r"\breduc(?:e|ed|ing) (?:our )?prices?\b", r"\bprice declines?\b",
        r"\bdiscounting\b", r"\bpromotional (?:activity|pricing)\b",
        r"\bpricing pressure\b", r"\bprice deflation\b",
    ]),
    "input_cost_up": tuple(re.compile(x, re.I) for x in [
        r"\bhigher (?:input|material|commodity|freight|transportation|labor|labour) costs?\b",
        r"\b(?:input|cost|raw material|commodity|freight|labor|labour|wage) inflation\b",
        r"\brising (?:input|material|commodity|freight|labor|labour) costs?\b",
        r"\bcost pressures?\b", r"\binflationary pressures?\b",
        r"\bincrease(?:d|s|ing)? (?:input|material|commodity|freight|labor|labour) costs?\b",
    ]),
    "input_cost_down": tuple(re.compile(x, re.I) for x in [
        r"\blower (?:input|material|commodity|freight|transportation|labor|labour) costs?\b",
        r"\b(?:input|cost|raw material|commodity|freight) deflation\b",
        r"\bdeclin(?:e|ed|ing) (?:input|material|commodity|freight) costs?\b",
        r"\beasing (?:input|material|commodity|freight) costs?\b",
        r"\bcost pressures? (?:eased|easing|moderated|subsided)\b",
    ]),
    "wage_up": tuple(re.compile(x, re.I) for x in [
        r"\bwage inflation\b", r"\bhigher (?:wages|labor costs?|labour costs?|compensation)\b",
        r"\brising (?:wages|labor costs?|labour costs?|compensation)\b",
        r"\bpay increases?\b", r"\btight labor market\b",
    ]),
    "wage_down": tuple(re.compile(x, re.I) for x in [
        r"\bwage (?:pressure|inflation) (?:eased|easing|moderated|slowed)\b",
        r"\blower (?:wages|labor costs?|labour costs?)\b", r"\blabor costs? declined\b",
    ]),
    "demand_up": tuple(re.compile(x, re.I) for x in [
        r"\bstrong(?:er)? demand\b", r"\bdemand (?:increased|improved|accelerated|remains strong)\b",
        r"\brobust demand\b", r"\bhealthy demand\b", r"\borders? (?:grew|increased|accelerated)\b",
    ]),
    "demand_down": tuple(re.compile(x, re.I) for x in [
        r"\bweak(?:er|ening)? demand\b", r"\bdemand (?:declined|slowed|softened|remains weak)\b",
        r"\bsoft demand\b", r"\bdemand pressure\b", r"\borders? (?:declined|slowed|decreased)\b",
    ]),
    "supply_tight": tuple(re.compile(x, re.I) for x in [
        r"\bsupply chain (?:constraint|constraints|disruption|disruptions|bottleneck|bottlenecks)\b",
        r"\bshortages?\b", r"\bcapacity constraints?\b", r"\blong(?:er)? lead times?\b",
        r"\bconstrained supply\b", r"\blogistics disruptions?\b",
    ]),
    "supply_ease": tuple(re.compile(x, re.I) for x in [
        r"\bsupply chain (?:improved|improving|normalized|normalizing|eased|easing)\b",
        r"\bshortages? (?:eased|easing|improved|abated)\b", r"\bshorter lead times?\b",
        r"\blogistics (?:improved|normalizing|normalized)\b",
    ]),
    "inventory_excess": tuple(re.compile(x, re.I) for x in [
        r"\bexcess inventor(?:y|ies)\b", r"\belevated inventor(?:y|ies)\b",
        r"\binventory correction\b", r"\bdestocking\b", r"\binventory overhang\b",
    ]),
    "inventory_short": tuple(re.compile(x, re.I) for x in [
        r"\blow inventor(?:y|ies)\b", r"\binventory shortages?\b",
        r"\btight inventor(?:y|ies)\b", r"\bunderstocked\b",
    ]),
    "margin_pressure": tuple(re.compile(x, re.I) for x in [
        r"\bmargin pressure\b", r"\bmargins? (?:declined|compressed|contracted)\b",
        r"\bcosts? (?:outpaced|exceeded) pricing\b", r"\bprice[- ]cost negative\b",
    ]),
    "pricing_absorption": tuple(re.compile(x, re.I) for x in [
        r"\bpricing (?:offset|covered|more than offset) (?:cost|inflation)\b",
        r"\bprice[- ]cost positive\b", r"\bpass[- ]through\b", r"\brecapture(?:d|ing)? inflation\b",
    ]),
    "uncertainty": tuple(re.compile(x, re.I) for x in [
        r"\buncertain(?:ty)?\b", r"\blimited visibility\b", r"\bvolatile environment\b",
        r"\bdifficult to predict\b", r"\bwide range of outcomes\b",
    ]),
}

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")


@dataclass
class CallFeatures:
    row: dict[str, Any]
    hash_vector: np.ndarray


def clean_text(row: dict[str, Any]) -> str:
    structured = row.get("structured_content")
    if isinstance(structured, list) and structured:
        parts: list[str] = []
        for segment in structured:
            if not isinstance(segment, dict):
                continue
            speaker = str(segment.get("speaker") or "").strip().lower()
            if speaker == "operator" or speaker.startswith("operator "):
                continue
            text = str(segment.get("text") or "").strip()
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(row.get("content") or "")


def iter_sentences(text: str) -> Iterable[str]:
    for sentence in SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if 20 <= len(sentence) <= 2500:
            yield sentence


def count_pattern_hits(sentence: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    return sum(1 for pattern in patterns if pattern.search(sentence))


def extract_call(row: dict[str, Any], vectorizer: HashingVectorizer) -> CallFeatures | None:
    text = clean_text(row)
    if not text.strip():
        return None
    words = WORD_RE.findall(text)
    n_words = len(words)
    if n_words < 50:
        return None

    raw = {name: 0 for name in PATTERNS}
    forward = {name: 0 for name in PATTERNS}
    for sentence in iter_sentences(text):
        is_forward = bool(FORWARD_RE.search(sentence))
        for name, patterns in PATTERNS.items():
            hits = count_pattern_hits(sentence, patterns)
            raw[name] += hits
            if is_forward:
                forward[name] += hits

    scale = 10_000.0 / n_words
    call_date = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.isna(call_date):
        return None
    month = call_date.to_period("M").to_timestamp("M")
    out: dict[str, Any] = {
        "symbol": str(row.get("symbol") or "").strip().upper(),
        "company_name": str(row.get("company_name") or "").strip(),
        "company_id": row.get("company_id"),
        "call_datetime": call_date.isoformat(),
        "month": month.date().isoformat(),
        "year": row.get("year"),
        "quarter": row.get("quarter"),
        "n_words": n_words,
        "content_sha256": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
    }
    for name in PATTERNS:
        out[f"{name}_hits"] = raw[name]
        out[f"{name}_per10k"] = raw[name] * scale
        out[f"{name}_forward_hits"] = forward[name]
        out[f"{name}_forward_per10k"] = forward[name] * scale

    def balance(up: str, down: str, suffix: str = "hits") -> float:
        a, b = float(out[f"{up}_{suffix}"]), float(out[f"{down}_{suffix}"])
        return (a - b) / (a + b + 1.0)

    out["price_balance"] = balance("final_price_up", "final_price_down")
    out["cost_balance"] = balance("input_cost_up", "input_cost_down")
    out["wage_balance"] = balance("wage_up", "wage_down")
    out["demand_balance"] = balance("demand_up", "demand_down")
    out["supply_balance"] = balance("supply_tight", "supply_ease")
    out["inventory_balance"] = balance("inventory_short", "inventory_excess")
    out["forward_price_balance"] = balance(
        "final_price_up", "final_price_down", "forward_hits"
    )
    out["forward_cost_balance"] = balance(
        "input_cost_up", "input_cost_down", "forward_hits"
    )

    # A fixed, leakage-free representation. Monthly means are exported; raw
    # transcript text and call-level vectors are not stored.
    vec = vectorizer.transform([text]).toarray()[0].astype(np.float64)
    return CallFeatures(out, vec)


def aggregate_monthly(calls: pd.DataFrame, hash_sums: dict[str, np.ndarray], hash_counts: dict[str, int]) -> pd.DataFrame:
    numeric_cols = [
        c for c in calls.columns
        if c.endswith("_per10k") or c.endswith("_balance") or c in {
            "price_balance", "cost_balance", "wage_balance", "demand_balance",
            "supply_balance", "inventory_balance", "forward_price_balance",
            "forward_cost_balance",
        }
    ]
    rows: list[dict[str, Any]] = []
    for month, group in calls.groupby("month", sort=True):
        item: dict[str, Any] = {
            "month": month,
            "n_calls": int(len(group)),
            "n_companies": int(group["symbol"].nunique()),
            "total_words": int(group["n_words"].sum()),
        }
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            item[f"{col}_mean"] = float(values.mean())
            item[f"{col}_median"] = float(values.median())
            item[f"{col}_std"] = float(values.std(ddof=0))
            item[f"{col}_p25"] = float(values.quantile(0.25))
            item[f"{col}_p75"] = float(values.quantile(0.75))
        for col in ["price_balance", "cost_balance", "demand_balance", "supply_balance", "wage_balance"]:
            item[f"{col}_positive_share"] = float((group[col] > 0).mean())
            item[f"{col}_negative_share"] = float((group[col] < 0).mean())
        vec = hash_sums[month] / max(hash_counts[month], 1)
        for j, value in enumerate(vec):
            item[f"hash_{j:03d}"] = float(value)
        rows.append(item)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/price_net/output")
    parser.add_argument("--hash-features", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vectorizer = HashingVectorizer(
        n_features=args.hash_features,
        alternate_sign=True,
        norm="l2",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    info = HfApi().dataset_info(DATASET_ID)
    stream = load_dataset(DATASET_ID, split="train", streaming=True)
    seen: set[tuple[str, str, str]] = set()
    records: list[dict[str, Any]] = []
    hash_sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(args.hash_features, dtype=np.float64))
    hash_counts: dict[str, int] = defaultdict(int)
    skipped_duplicates = 0
    skipped_empty = 0

    for i, source_row in enumerate(stream):
        if args.max_rows and i >= args.max_rows:
            break
        feature = extract_call(dict(source_row), vectorizer)
        if feature is None:
            skipped_empty += 1
            continue
        key = (
            feature.row["symbol"], feature.row["call_datetime"], feature.row["content_sha256"]
        )
        if key in seen:
            skipped_duplicates += 1
            continue
        seen.add(key)
        records.append(feature.row)
        month = feature.row["month"]
        hash_sums[month] += feature.hash_vector
        hash_counts[month] += 1
        if len(records) % 1000 == 0:
            print(f"processed {len(records):,} unique calls", flush=True)

    calls = pd.DataFrame(records).sort_values(["call_datetime", "symbol"]).reset_index(drop=True)
    monthly = aggregate_monthly(calls, hash_sums, hash_counts)
    calls_path = output / "earnings_call_price_features.csv.gz"
    monthly_path = output / "earnings_call_monthly_features.csv"
    calls.to_csv(calls_path, index=False, compression="gzip")
    monthly.to_csv(monthly_path, index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": DATASET_ID,
        "source_revision": info.sha,
        "source_license": "MIT (per dataset card)",
        "raw_text_redistributed": False,
        "unique_calls": int(len(calls)),
        "unique_symbols": int(calls["symbol"].nunique()),
        "first_call": calls["call_datetime"].min(),
        "last_call": calls["call_datetime"].max(),
        "monthly_rows": int(len(monthly)),
        "hash_features": args.hash_features,
        "skipped_duplicates": skipped_duplicates,
        "skipped_empty_or_short": skipped_empty,
        "method": "contextual economic phrase counts plus fixed HashingVectorizer monthly means",
        "warning": "Transparent stage-1 text baseline; not a claim of LLM-equivalent extraction.",
    }
    (output / "earnings_call_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
