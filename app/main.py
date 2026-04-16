import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.agents.orchestrator import run_pipeline
from app.schemas import ChatResponse

app = FastAPI(title="Stock Market Analyst AI")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve artifacts (generated charts) for download
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")


class ChatReq(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/debug-collect")
def debug_collect():
    """Test just the collector stage."""
    from app.agents.orchestrator import run_collector
    try:
        result = run_collector("Which S&P 500 sector had the highest average return in 2023?")
        if result is None:
            return {"error": "run_collector returned None"}
        return {
            "source_type":       result.data_source.source_type,
            "row_count":         result.data_source.row_count,
            "sql_query":         result.data_source.sql_query,
            "columns":           result.columns,
            "collection_notes":  result.collection_notes,
            "raw_json_length":   len(result.raw_json),
            "raw_json_preview":  result.raw_json[:300],
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/debug-stats")
def debug_stats():
    """Test run_stats directly with a known good payload."""
    from app.tools.stats_tool import run_stats
    import json
    sample = json.dumps([
        {"sector": "Technology",    "avg_daily_return": 0.00124, "annualised_return": 0.367, "annualised_volatility": 0.28},
        {"sector": "Healthcare",    "avg_daily_return": -0.00002,"annualised_return": -0.004,"annualised_volatility": 0.19},
        {"sector": "Financials",    "avg_daily_return": 0.00044, "annualised_return": 0.116, "annualised_volatility": 0.22},
    ])
    result = run_stats(sample, "auto")
    return result


@app.get("/db-check")
def db_check():
    """Quick diagnostic: confirms DB exists and has data."""
    import duckdb
    from pathlib import Path
    db_path = Path(__file__).resolve().parents[1] / "data" / "market.duckdb"
    if not db_path.exists():
        return {"error": f"DB not found at {db_path}"}
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        result = {}
        for tbl in ["ohlcv", "earnings", "sector_meta"]:
            n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            result[tbl] = n
        date_range = con.execute(
            "SELECT MIN(date), MAX(date) FROM ohlcv"
        ).fetchone()
        result["ohlcv_date_range"] = {
            "min": str(date_range[0]),
            "max": str(date_range[1])
        }
        con.close()
        return {"status": "ok", "tables": result}
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatReq):
    text = (req.text or "").strip()
    if not text:
        return ChatResponse(
            answer="Please enter a question about US stock market data.",
            backstop="empty",
        )
    return run_pipeline(text)