"""
tools/sql_tool.py — Text2SQL + DuckDB executor
"""

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

# Resolve DB path relative to this file, works on Windows and Linux
DB_PATH = str(Path(__file__).resolve().parents[2] / "data" / "market.duckdb")
log.info("sql_tool: DB_PATH resolved to %s", DB_PATH)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
MODEL   = "gemini-2.0-flash-001"

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

Sectors present: Technology, Healthcare, Financials, Consumer Discretionary,
Consumer Staples, Industrials, Energy, Utilities, Real Estate, Materials,
Communication Services

Example useful calculations:
- Daily return: (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) / LAG(close) OVER ...
- Annualised volatility: STDDEV(daily_return) * SQRT(252)
- Sector avg return: AVG over tickers in that sector
"""

TEXT2SQL_SYSTEM = f"""
You are a DuckDB SQL expert. Given a natural-language analytics question about
US stock market data, write a single valid DuckDB SQL query.

Rules:
- Return ONLY the raw SQL, no markdown fences, no explanation.
- Use only tables and columns defined in the schema below.
- Always LIMIT results to at most 500 rows unless the question explicitly needs more.
- Prefer CTEs for readability.
- For volatility use: STDDEV(daily_return) * SQRT(252) pattern.
- For returns use LAG window functions.
- Cast dates with CAST('2023-01-01' AS DATE) syntax.

{SCHEMA_CONTEXT}
"""


def generate_sql(question: str) -> str:
    """Use Gemini to translate a natural-language question into DuckDB SQL."""
    response = _client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=TEXT2SQL_SYSTEM,
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )
    sql = (response.text or "").strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.I).strip()
    sql = re.sub(r"```$", "", sql).strip()
    log.info("generate_sql: produced SQL:\n%s", sql)
    return sql


def execute_sql(sql: str) -> tuple[pd.DataFrame, str]:
    """Execute SQL against the local DuckDB file."""
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
    """
    Full pipeline: question -> SQL -> execute -> return result dict.
    Used by the Collector agent as a tool.
    """
    sql = generate_sql(question)
    df, err = execute_sql(sql)

    if err:
        log.warning("run_text2sql: first attempt failed (%s), retrying...", err)
        retry_prompt = (
            f"The following SQL failed with error: {err}\n\n"
            f"Original SQL:\n{sql}\n\n"
            f"Rewrite the SQL to fix the error. Return ONLY the corrected SQL."
        )
        sql = generate_sql(retry_prompt)
        df, err = execute_sql(sql)

    if err or df.empty:
        log.error("run_text2sql: final result empty or error: %s", err)
        return {
            "success":   False,
            "sql":       sql,
            "error":     err or "Query returned no rows.",
            "row_count": 0,
            "columns":   [],
            "preview":   [],
            "json_data": "[]",
        }

    # Hard cap: truncate to 100 rows to keep json_data safely within LLM context
    if len(df) > 100:
        log.warning("run_text2sql: truncating %d rows to 100", len(df))
        df = df.head(100)

    json_data = df.to_json(orient="records", date_format="iso")

    # Safety cap on json_data string size (max ~50KB)
    if len(json_data) > 50_000:
        log.warning("run_text2sql: json_data too large (%d chars), truncating", len(json_data))
        # Re-truncate to fewer rows until it fits
        for n in [50, 20, 10]:
            json_data = df.head(n).to_json(orient="records", date_format="iso")
            if len(json_data) <= 50_000:
                df = df.head(n)
                break

    result = {
        "success":   True,
        "sql":       sql,
        "error":     "",
        "row_count": len(df),
        "columns":   list(df.columns),
        "preview":   df.head(5).to_dict(orient="records"),
        "json_data": json_data,
    }
    log.info("run_text2sql: success, %d rows, json_data=%d chars", len(df), len(json_data))
    return result