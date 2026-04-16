import os
import time
import duckdb
import pandas as pd
import yfinance as yf

DB_PATH = os.path.join(os.path.dirname(__file__), "market.duckdb")

TICKERS = {
    "Technology":             ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Healthcare":             ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
    "Financials":             ["JPM", "BAC", "WFC", "GS", "MS"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "Consumer Staples":       ["PG", "KO", "PEP", "WMT", "COST"],
    "Industrials":            ["GE", "CAT", "HON", "UPS", "BA"],
    "Energy":                 ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Utilities":              ["NEE", "DUK", "SO", "AEP", "EXC"],
    "Real Estate":            ["PLD", "AMT", "EQIX", "SPG", "PSA"],
    "Materials":              ["LIN", "APD", "SHW", "FCX", "NEM"],
    "Communication Services": ["VZ", "T", "NFLX", "DIS", "CMCSA"],
}

FLAT_TICKERS = [t for tickers in TICKERS.values() for t in tickers]
SECTOR_MAP   = {t: s for s, tickers in TICKERS.items() for t in tickers}


def build_ohlcv(con: duckdb.DuckDBPyConnection) -> None:
    print("Downloading OHLCV (5 years)…")
    frames = []
    for ticker in FLAT_TICKERS:
        try:
            df = yf.download(ticker, period="5y", interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"  SKIP {ticker}: no data")
                continue
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten: take the first level (Price) and drop ticker level
                df.columns = [col[0] if isinstance(col, tuple) else col
                               for col in df.columns]
            df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
            print(f"  {ticker} raw columns: {list(df.columns)}")
            df["ticker"] = ticker
            df["sector"] = SECTOR_MAP[ticker]
            if "adj_close" in df.columns:
                df = df.rename(columns={"adj_close": "close"})
            keep = ["date", "ticker", "sector", "open", "high", "low",
                    "close", "volume"]
            missing = [c for c in keep if c not in df.columns]
            if missing:
                print(f"  SKIP {ticker}: missing columns {missing}, have {list(df.columns)}")
                continue
            df = df[keep]
            frames.append(df)
            print(f"  OK  {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"  ERR {ticker}: {e}")
        time.sleep(0.2)

    if not frames:
        raise RuntimeError("No OHLCV data downloaded — check network / yfinance.")

    ohlcv = pd.concat(frames, ignore_index=True)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date

    con.execute("DROP TABLE IF EXISTS ohlcv")
    con.register("ohlcv_df", ohlcv)
    con.execute("""
        CREATE TABLE ohlcv AS SELECT
            CAST(date AS DATE) AS date,
            CAST(ticker AS VARCHAR) AS ticker,
            CAST(sector AS VARCHAR) AS sector,
            CAST(open   AS DOUBLE)  AS open,
            CAST(high   AS DOUBLE)  AS high,
            CAST(low    AS DOUBLE)  AS low,
            CAST(close  AS DOUBLE)  AS close,
            CAST(volume AS BIGINT)  AS volume
        FROM ohlcv_df
    """)
    con.unregister("ohlcv_df")
    actual = con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    print(f"  ohlcv table: {actual:,} rows")


def build_earnings(con: duckdb.DuckDBPyConnection) -> None:
    print("Downloading earnings…")
    rows = []
    for ticker in FLAT_TICKERS:
        try:
            tk   = yf.Ticker(ticker)
            hist = tk.earnings_history
            if hist is None or hist.empty:
                print(f"  SKIP {ticker}: no earnings history")
                continue
            hist = hist.reset_index()
            hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
            for _, r in hist.iterrows():
                rows.append({
                    "ticker":            ticker,
                    "sector":            SECTOR_MAP[ticker],
                    "quarter":           str(r.get("quarter", "")),
                    "eps_estimate":      float(r["epsestimate"]) if "epsestimate" in r and pd.notna(r["epsestimate"]) else None,
                    "eps_actual":        float(r["epsactual"])   if "epsactual"   in r and pd.notna(r["epsactual"])   else None,
                    "surprise_percent":  float(r["surprisepercent"]) if "surprisepercent" in r and pd.notna(r["surprisepercent"]) else None,
                })
            print(f"  OK  {ticker}: {len(hist)} quarters")
        except Exception as e:
            print(f"  ERR {ticker}: {e}")
        time.sleep(0.3)

    if not rows:
        print("  WARNING: no earnings data fetched; table will be empty.")

    earnings = pd.DataFrame(rows)
    con.execute("DROP TABLE IF EXISTS earnings")
    if not earnings.empty:
        con.register("earnings_df", earnings)
        con.execute("""
            CREATE TABLE earnings AS SELECT
                CAST(ticker           AS VARCHAR) AS ticker,
                CAST(sector           AS VARCHAR) AS sector,
                CAST(quarter          AS VARCHAR) AS quarter,
                CAST(eps_estimate     AS DOUBLE)  AS eps_estimate,
                CAST(eps_actual       AS DOUBLE)  AS eps_actual,
                CAST(surprise_percent AS DOUBLE)  AS surprise_percent
            FROM earnings_df
        """)
        con.unregister("earnings_df")
    else:
        con.execute("""
            CREATE TABLE earnings (
                ticker VARCHAR, sector VARCHAR, quarter VARCHAR,
                eps_estimate DOUBLE, eps_actual DOUBLE, surprise_percent DOUBLE
            )
        """)
    actual = con.execute("SELECT COUNT(*) FROM earnings").fetchone()[0]
    print(f"  earnings table: {actual:,} rows")


def build_sector_meta(con: duckdb.DuckDBPyConnection) -> None:
    rows = [{"ticker": t, "sector": s} for t, s in SECTOR_MAP.items()]
    df   = pd.DataFrame(rows)
    con.execute("DROP TABLE IF EXISTS sector_meta")
    con.register("sector_meta_df", df)
    con.execute("""
        CREATE TABLE sector_meta AS SELECT
            CAST(ticker AS VARCHAR) AS ticker,
            CAST(sector AS VARCHAR) AS sector
        FROM sector_meta_df
    """)
    con.unregister("sector_meta_df")
    actual = con.execute("SELECT COUNT(*) FROM sector_meta").fetchone()[0]
    print(f"  sector_meta table: {actual} rows")


def main() -> None:
    print(f"Building DuckDB at: {DB_PATH}")
    con = duckdb.connect(DB_PATH)
    build_ohlcv(con)
    build_earnings(con)
    build_sector_meta(con)

    print("\nSchema summary:")
    for tbl in ["ohlcv", "earnings", "sector_meta"]:
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl}: {n:,} rows")

    con.close()
    print("\nDone. market.duckdb is ready.")


if __name__ == "__main__":
    main()