"""
tools/api_tool.py — Live yfinance data fetcher (second retrieval method)

Called when the user asks about:
- Data within the last 30 days (may not be in the DB yet)
- A specific ticker not in the DB
- Current price / latest news
"""

from __future__ import annotations
import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


def fetch_live_ohlcv(tickers: list[str], period: str = "1mo") -> dict:
    """
    Fetch recent OHLCV for given tickers from yfinance.
    period: '5d', '1mo', '3mo', '6mo', '1y'
    """
    try:
        data = yf.download(
            tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker" if len(tickers) > 1 else "column",
        )
        if data.empty:
            return {"success": False, "error": "No data returned.", "rows": []}

        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ["_".join(filter(None, c)).strip() for c in data.columns]
        data.columns = [c.lower().replace(" ", "_") for c in data.columns]

        rows = data.to_dict(orient="records")
        # Convert dates to ISO strings
        for r in rows:
            for k, v in r.items():
                if isinstance(v, (datetime, pd.Timestamp)):
                    r[k] = v.isoformat()

        return {
            "success":  True,
            "tickers":  tickers,
            "period":   period,
            "row_count": len(rows),
            "columns":  list(data.columns),
            "rows":     rows,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "rows": []}


def fetch_ticker_info(ticker: str) -> dict:
    """Fetch summary info (market cap, PE, sector, etc.) for a single ticker."""
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}
        keep = [
            "symbol", "shortName", "sector", "industry",
            "marketCap", "trailingPE", "forwardPE",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "currentPrice", "targetMeanPrice",
            "revenueGrowth", "earningsGrowth",
            "dividendYield", "beta",
        ]
        return {k: info.get(k) for k in keep}
    except Exception as e:
        return {"error": str(e)}


def fetch_recent_earnings(tickers: list[str]) -> dict:
    """Fetch the most recent earnings history for given tickers."""
    results = {}
    for ticker in tickers:
        try:
            tk   = yf.Ticker(ticker)
            hist = tk.earnings_history
            if hist is not None and not hist.empty:
                hist = hist.reset_index()
                hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
                results[ticker] = hist.tail(8).to_dict(orient="records")
            else:
                results[ticker] = []
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results


def run_api_fetch(tickers: list[str], fetch_type: str = "ohlcv", period: str = "1mo") -> dict:
    """
    Unified entry point for the Collector agent.
    fetch_type: 'ohlcv' | 'info' | 'earnings'
    """
    if fetch_type == "ohlcv":
        return fetch_live_ohlcv(tickers, period)
    elif fetch_type == "info":
        return {t: fetch_ticker_info(t) for t in tickers}
    elif fetch_type == "earnings":
        return fetch_recent_earnings(tickers)
    else:
        return {"error": f"Unknown fetch_type: {fetch_type}"}