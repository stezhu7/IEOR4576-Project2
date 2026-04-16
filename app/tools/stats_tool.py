from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd
from typing import Any


def _safe_float(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(float(v), 4)
    return v


def compute_return_stats(df: pd.DataFrame) -> list[dict]:
    results = []
    if "close" not in df.columns or "date" not in df.columns:
        return results

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"] if "ticker" in df.columns else ["date"])

    group_col = "ticker" if "ticker" in df.columns else None

    if group_col:
        df["daily_return"] = df.groupby(group_col)["close"].pct_change()
    else:
        df["daily_return"] = df["close"].pct_change()

    for name, grp in (df.groupby(group_col) if group_col else [("overall", df)]):
        ret = grp["daily_return"].dropna()
        if len(ret) < 5:
            continue
        ann_return = _safe_float((1 + ret.mean()) ** 252 - 1)
        ann_vol    = _safe_float(ret.std() * math.sqrt(252))
        sharpe     = _safe_float(ann_return / ann_vol) if ann_vol else None
        results.append({
            "metric": f"{name} annualised_return",
            "value":  ann_return,
            "unit":   "ratio",
        })
        results.append({
            "metric": f"{name} annualised_volatility",
            "value":  ann_vol,
            "unit":   "ratio",
        })
        if sharpe is not None:
            results.append({
                "metric": f"{name} sharpe_ratio",
                "value":  sharpe,
                "unit":   "ratio",
            })

    return results


def compute_sector_stats(df: pd.DataFrame) -> list[dict]:
    results = []
    if "sector" not in df.columns or "close" not in df.columns:
        return results

    df = df.copy()
    df["date"]         = pd.to_datetime(df["date"])
    df["daily_return"] = df.groupby("ticker")["close"].pct_change() if "ticker" in df.columns else df["close"].pct_change()

    for sector, grp in df.groupby("sector"):
        ret = grp["daily_return"].dropna()
        if len(ret) < 5:
            continue
        results.append({
            "metric": f"{sector} avg_daily_return",
            "value":  _safe_float(ret.mean()),
            "unit":   "ratio",
        })
        results.append({
            "metric": f"{sector} volatility",
            "value":  _safe_float(ret.std() * math.sqrt(252)),
            "unit":   "annualised",
        })

    return results


def compute_earnings_stats(df: pd.DataFrame) -> list[dict]:
    results = []
    if "surprise_percent" not in df.columns:
        return results

    surp = df["surprise_percent"].dropna()
    if surp.empty:
        return results

    results.append({"metric": "mean_earnings_surprise", "value": _safe_float(surp.mean()), "unit": "%"})
    results.append({"metric": "median_earnings_surprise", "value": _safe_float(surp.median()), "unit": "%"})
    results.append({"metric": "pct_beats", "value": _safe_float((surp > 0).mean() * 100), "unit": "%"})
    results.append({"metric": "pct_misses", "value": _safe_float((surp < 0).mean() * 100), "unit": "%"})

    if "ticker" in df.columns:
        best  = df.groupby("ticker")["surprise_percent"].mean().idxmax()
        worst = df.groupby("ticker")["surprise_percent"].mean().idxmin()
        results.append({"metric": "best_earnings_beater", "value": best, "unit": "ticker"})
        results.append({"metric": "worst_earnings_misser", "value": worst, "unit": "ticker"})

    return results


def compute_correlation(df: pd.DataFrame, col_a: str, col_b: str) -> list[dict]:
    if col_a not in df.columns or col_b not in df.columns:
        return []
    corr = df[[col_a, col_b]].dropna().corr().iloc[0, 1]
    return [{"metric": f"pearson_corr({col_a},{col_b})", "value": _safe_float(corr), "unit": "r"}]


def detect_anomalies(df: pd.DataFrame) -> list[str]:
    anomalies = []
    if "close" not in df.columns:
        return anomalies

    df = df.copy()
    if "ticker" in df.columns:
        df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    else:
        df["daily_return"] = df["close"].pct_change()

    ret    = df["daily_return"].dropna()
    mean_r = ret.mean()
    std_r  = ret.std()
    if std_r == 0:
        return anomalies

    outliers = df[abs(df["daily_return"] - mean_r) > 3 * std_r]
    for _, row in outliers.head(5).iterrows():
        ticker = row.get("ticker", "N/A")
        date   = str(row.get("date", "N/A"))
        ret_v  = row.get("daily_return", float("nan"))
        anomalies.append(
            f"{ticker} on {date}: daily return {ret_v:.2%} (>{3}σ outlier)"
        )

    return anomalies


import logging as _logging
_log = _logging.getLogger(__name__)

def run_stats(json_data: str, analysis_type: str = "auto") -> dict:
    _log.info("run_stats: json_data length=%d, analysis_type=%s", len(json_data or ""), analysis_type)
    try:
        records = json.loads(json_data)
        df      = pd.DataFrame(records)
        _log.info("run_stats: parsed %d rows, columns=%s", len(df), list(df.columns))
    except Exception as e:
        _log.error("run_stats: JSON parse failed: %s | first 200 chars: %r", e, (json_data or "")[:200])
        return {"stats": [], "anomalies": [], "error": str(e)}

    if df.empty:
        _log.warning("run_stats: DataFrame is empty")
        return {"stats": [], "anomalies": [], "error": "Empty dataset"}

    stats = []
    # Convert date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Detect data type
    has_raw_prices = "close" in df.columns
    has_precomputed = any(
        c for c in df.columns
        if any(k in c.lower() for k in ["return", "volatility", "surprise", "eps", "sharpe"])
    )
    has_market_info = any(
        c for c in df.columns
        if any(k in c.lower() for k in ["marketcap", "market_cap", "pe", "beta",
                                         "currentprice", "current_price", "targetmean"])
    )

    if not has_raw_prices and (has_precomputed or has_market_info):
        _log.info("run_stats: detected pre-aggregated/info data, using precomputed handler")
        stats = compute_precomputed_stats(df)
    else:
        if analysis_type in ("auto", "returns"):
            stats += compute_return_stats(df)

        if analysis_type in ("auto", "sector") and "sector" in df.columns:
            stats += compute_sector_stats(df)

        if analysis_type in ("auto", "earnings") and "surprise_percent" in df.columns:
            stats += compute_earnings_stats(df)

    anomalies = detect_anomalies(df) if has_raw_prices else []

    _log.info("run_stats: produced %d stats", len(stats))
    return {
        "stats":      stats,
        "anomalies":  anomalies,
        "row_count":  len(df),
        "columns":    list(df.columns),
    }


def compute_precomputed_stats(df: pd.DataFrame) -> list[dict]:
    results = []
    label_col = None
    for c in ["sector", "ticker", "name"]:
        if c in df.columns:
            label_col = c
            break

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for _, row in df.iterrows():
        label = str(row[label_col]) if label_col else "row"
        for col in numeric_cols:
            val = row[col]
            if pd.isna(val):
                continue
            # Detect unit
            if any(k in col for k in ["return", "yield", "growth", "surprise"]):
                unit = "ratio" if abs(val) < 5 else "%"
            elif any(k in col for k in ["volatility", "std", "vol"]):
                unit = "annualised"
            elif any(k in col for k in ["volume", "count", "observations"]):
                unit = "count"
            else:
                unit = None
            results.append({
                "metric": f"{label} {col}",
                "value":  _safe_float(val),
                "unit":   unit,
            })
    return results