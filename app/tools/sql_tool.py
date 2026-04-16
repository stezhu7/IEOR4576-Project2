from __future__ import annotations
import json
import logging
import os
import re
from pathlib import Path

import duckdb
import pandas as pd
from google import genai
from google.genai import types

log = logging.getLogger(__name__)

DB_PATH = str(Path(__file__).resolve().parents[2] / "data" / "market.duckdb")
log.info("sql_tool: DB_PATH resolved to %s", DB_PATH)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
MODEL   = "gemini-2.5-flash"

# Columns that exist in the DB
DB_COLUMNS = {
    "ohlcv":       {"date", "ticker", "sector", "open", "high", "low", "close", "volume"},
    "earnings":    {"ticker", "sector", "quarter", "eps_estimate", "eps_actual", "surprise_percent"},
    "sector_meta": {"ticker", "sector"},
}
ALL_DB_COLUMNS = {c for cols in DB_COLUMNS.values() for c in cols}

# Columns that only exist in the live API (yfinance info)
API_ONLY_FIELDS = {
    "marketcap", "market_cap", "trailingpe", "forwardpe", "pe", "pricetosales",
    "currentprice", "current_price", "targetmeanprice", "targetprice",
    "dividendyield", "dividend_yield", "beta", "earningsgrowth", "revenuegrowth",
    "shortname", "longname", "industry", "fullTimeEmployees",
}

SCHEMA_CONTEXT = """
DuckDB database: market.duckdb

Tables:

1. ohlcv
   - date     DATE        trading date
   - ticker   VARCHAR     stock symbol (e.g. 'AAPL')
   - sector   VARCHAR     GICS sector name
   - open     DOUBLE      opening price
   - high     DOUBLE      daily high
   - low      DOUBLE      daily low
   - close    DOUBLE      closing price
   - volume   BIGINT      shares traded

2. earnings
   - ticker           VARCHAR
   - sector           VARCHAR
   - quarter          VARCHAR    e.g. '2023Q4'
   - eps_estimate     DOUBLE     analyst consensus EPS estimate
   - eps_actual       DOUBLE     reported EPS
   - surprise_percent DOUBLE     (actual-estimate)/|estimate| * 100

3. sector_meta
   - ticker  VARCHAR PRIMARY KEY
   - sector  VARCHAR

This database contains ONLY price/volume/earnings data.
It does NOT contain: market cap, P/E ratio, current price, beta,
dividend yield, revenue growth, or any other fundamental/live data.

Sectors: Technology, Healthcare, Financials, Consumer Discretionary,
Consumer Staples, Industrials, Energy, Utilities, Real Estate, Materials,
Communication Services

Tickers covered: AAPL MSFT NVDA GOOGL META JNJ UNH PFE ABBV MRK
JPM BAC WFC GS MS AMZN TSLA HD MCD NKE PG KO PEP WMT COST
GE CAT HON UPS BA XOM CVX COP SLB EOG NEE DUK SO AEP EXC
PLD AMT EQIX SPG PSA LIN APD SHW FCX NEM VZ T NFLX DIS CMCSA
"""

TEXT2SQL_SYSTEM = f"""
You are a DuckDB SQL expert for a US stock market database.

IMPORTANT ROUTING RULE:
If the question asks for data that is NOT in the database — such as:
  market capitalisation, market cap, current stock price, P/E ratio,
  price-to-earnings, forward PE, beta, dividend yield, revenue growth,
  earnings growth, analyst target price, "right now", "current value",
  or any other live/fundamental metric —
then respond with ONLY this single word (no SQL, no explanation):
  NEEDS_API

Otherwise, write a single valid DuckDB SQL query following these rules:
- Return ONLY the raw SQL, no markdown fences, no explanation
- Use only the tables and columns defined in the schema
- Always aggregate results (GROUP BY, AVG, ROUND) — never raw row dumps
- NEVER use LIMIT 1 for comparison questions — return ALL rows
- For returns: use LAG window function over close prices
- For volatility: STDDEV(daily_return) * SQRT(252)
- Cast dates: CAST('2023-01-01' AS DATE)

{SCHEMA_CONTEXT}
"""


def generate_sql(question: str) -> str:
    response = _client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=TEXT2SQL_SYSTEM,
            temperature=0.0,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    sql = (response.text or "").strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.I).strip()
    sql = re.sub(r"```$", "", sql).strip()
    log.info("generate_sql: produced:\n%s", sql)
    return sql


def _references_api_only_fields(sql: str) -> bool:
    sql_lower = sql.lower()
    for field in API_ONLY_FIELDS:
        if field in sql_lower:
            log.warning("_references_api_only_fields: found API-only field '%s' in SQL", field)
            return True
    return False


def execute_sql(sql: str) -> tuple[pd.DataFrame, str]:
    # Strip LIMIT 1
    cleaned = re.sub(r"\bLIMIT\s+1\b\s*;?", "", sql, flags=re.IGNORECASE).strip()
    if cleaned != sql:
        log.warning("execute_sql: removed LIMIT 1 from SQL")
        sql = cleaned

    log.info("execute_sql: connecting to %s", DB_PATH)
    if not Path(DB_PATH).exists():
        err = f"DB file not found: {DB_PATH}"
        log.error(err)
        return pd.DataFrame(), err
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df  = con.execute(sql).df()
        con.close()
        log.info("execute_sql: returned %d rows, columns=%s", len(df), list(df.columns))
        return df, ""
    except Exception as e:
        log.error("execute_sql: failed: %s", e)
        return pd.DataFrame(), str(e)


def run_text2sql(question: str) -> dict:
    sql = generate_sql(question)

    # Check if model signalled that live API data is needed
    if sql.strip().upper() == "NEEDS_API" or _references_api_only_fields(sql):
        log.info("run_text2sql: routing to API — question needs live data")
        return {
            "success":   False,
            "needs_api": True,
            "sql":       "",
            "error":     "This question requires live data not in the database (e.g. market cap, current price, P/E). Use run_api_fetch instead.",
            "row_count": 0,
            "columns":   [],
            "preview":   [],
            "json_data": "[]",
        }

    df, err = execute_sql(sql)

    if err:
        log.warning("run_text2sql: first attempt failed (%s), retrying...", err)
        retry_prompt = (
            f"The following SQL failed with error: {err}\n\n"
            f"Original SQL:\n{sql}\n\n"
            f"Rewrite the SQL to fix the error. Return ONLY the corrected SQL."
        )
        sql = generate_sql(retry_prompt)

        # If retry also signals NEEDS_API
        if sql.strip().upper() == "NEEDS_API":
            return {
                "success":   False,
                "needs_api": True,
                "sql":       "",
                "error":     "This question requires live data not in the database.",
                "row_count": 0,
                "columns":   [],
                "preview":   [],
                "json_data": "[]",
            }

        df, err = execute_sql(sql)

    if err or df.empty:
        log.error("run_text2sql: final result empty or error: %s", err)
        return {
            "success":   False,
            "needs_api": False,
            "sql":       sql,
            "error":     err or "Query returned no rows.",
            "row_count": 0,
            "columns":   [],
            "preview":   [],
            "json_data": "[]",
        }

    # Hard cap: 100 rows max
    if len(df) > 100:
        log.warning("run_text2sql: truncating %d rows to 100", len(df))
        df = df.head(100)

    json_data = df.to_json(orient="records", date_format="iso")

    # 50KB size cap
    if len(json_data) > 50_000:
        for n in [50, 20, 10]:
            json_data = df.head(n).to_json(orient="records", date_format="iso")
            if len(json_data) <= 50_000:
                df = df.head(n)
                break

    result = {
        "success":   True,
        "needs_api": False,
        "sql":       sql,
        "error":     "",
        "row_count": len(df),
        "columns":   list(df.columns),
        "preview":   df.head(5).to_dict(orient="records"),
        "json_data": json_data,
    }
    log.info("run_text2sql: success, %d rows, json_data=%d chars", len(df), len(json_data))
    return result