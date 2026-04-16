import duckdb
con = duckdb.connect("data/market.duckdb")
print(con.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM ohlcv").fetchdf())
print(con.execute("SELECT sector, COUNT(*) FROM ohlcv GROUP BY sector").fetchdf())