"""
app/main.py — FastAPI entry point for the Stock Market Analyst AI
"""

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