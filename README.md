# Stock Market Analyst AI

A multi-agent system that performs the first three steps of a data analysis lifecycle — **Collect → EDA → Hypothesize** — over real US stock market data.

Ask questions like:
- *"Which S&P 500 sector had the highest average return in 2023?"*
- *"Compare XXX volatility over the last 2 years"*
- *"Is there a correlation between earnings surprise and next-day price movement?"*

---

## Live Demo

**Deployed URL:** `https://ieor4576-project2-git-7610618360.us-central1.run.app/`

---

## The Three Steps

### Step 1 — Collect (`app/agents/orchestrator.py` → `run_collector`)

The Collector agent retrieves real data at runtime using two distinct methods:

**Method A — Text2SQL against DuckDB** (`app/tools/sql_tool.py` → `run_text2sql`)

The agent translates the user's natural-language question into a DuckDB SQL query using Gemini (`generate_sql`), executes it against `data/market.duckdb`, and returns structured results. The database contains:
- `ohlcv` — ~69,000 rows of daily OHLCV prices for 50 S&P 500 tickers across 11 GICS sectors, covering 5 years
- `earnings` — 220 rows of quarterly EPS actual vs estimate with surprise percentages
- `sector_meta` — ticker-to-sector mapping for all 50 tickers

The dataset is far too large to load into context — the agent writes SQL dynamically based on the question asked.

**Method B — Live yfinance API** (`app/tools/api_tool.py` → `run_api_fetch`)

For queries involving recent data (last 30 days) or tickers not in the DB, the Collector calls the yfinance API directly at runtime to fetch live OHLCV, summary info, or recent earnings.

The Collector decides which source to use (SQL, API, or both) based on the question's time range and scope.

---

### Step 2 — EDA (`app/agents/orchestrator.py` → `run_eda`)

The EDA agent performs exploratory data analysis on the collected data before forming any hypothesis. It uses two tool calls:

**Tool 1 — Statistical aggregation** (`app/tools/stats_tool.py` → `run_stats`)

Computes annualised returns, volatility, Sharpe ratios, earnings surprise distributions, correlations, and anomaly detection (>3σ outliers) over the collected rows. Returns per-sector or per-ticker breakdowns, not generic summaries.

**Tool 2 — Data visualisation** (`app/tools/viz_tool.py` → `run_viz`)

Generates a matplotlib chart (bar for sector comparisons, line for time-series, scatter for correlations), saves it as a PNG artifact to `artifacts/`, and returns it as a base64 string for the frontend to render inline.

The EDA is dynamic — a question about sector returns produces a ranked bar chart; a question about correlations produces a scatter with regression line. The agent surfaces specific numbers (e.g. "Technology: 36.7% annualised return") that feed directly into the hypothesis.

---

### Step 3 — Hypothesize (`app/agents/orchestrator.py` → `run_hypothesis`)

The Hypothesis agent receives the EDA findings and writes a data-grounded analyst memo. It:
- Names the specific top and bottom performers with exact figures from the EDA
- Lists evidence points each citing a specific metric value
- Acknowledges caveats and limitations
- Assigns a confidence level (high / medium / low) based on evidence strength

The hypothesis is derived exclusively from the EDA output — the agent's prompt explicitly prohibits using general market knowledge from model weights.

---

## Architecture

```
User query
    │
    ▼
Orchestrator (semantic intent classification)
    │
    ├── OOS → redirect message
    ├── Safety → crisis response
    │
    └── In-scope
            │
            ▼
        Collector agent
        ├── run_text2sql  →  DuckDB (historical)
        └── run_api_fetch →  yfinance (live)
            │
            ▼
        EDA agent  ◄──── iterative refinement loop (max 3×)
        ├── run_stats  (aggregations, anomaly detection)
        └── run_viz    (bar / line / scatter chart)
            │
            ▼
        Hypothesis agent
        └── grounded narrative + evidence + caveats
            │
            ▼
        Frontend response
```

**Multi-agent pattern:** Orchestrator-handoff. The root orchestrator classifies intent and drives the pipeline imperatively, handing off to each sub-agent in sequence. Each agent has a distinct system prompt and responsibility. The orchestrator also runs an **iterative refinement loop** — if the EDA agent returns `sufficient=false`, the orchestrator re-invokes the collector with a narrowed query (up to `MAX_REFINEMENTS=3` attempts).

---

## Core Requirements

| Requirement | Implementation |
|---|---|
| Frontend | `static/index.html` — dark-themed chat UI with inline chart rendering, stat pills, pipeline progress indicator |
| Agent framework | `google-genai` (Vertex AI) — custom multi-turn agentic loop in `app/agents/orchestrator.py` → `_agentic_loop`, with 4 agents each defined by a distinct system prompt (ROUTE_SYSTEM, COLLECTOR_SYSTEM, EDA_SYSTEM, HYPOTHESIS_SYSTEM) |
| Tool calling | `run_text2sql`, `run_api_fetch`, `run_stats`, `run_viz` — all called at runtime via the agentic loop |
| Non-trivial dataset | 69,025 OHLCV rows + 220 earnings rows in DuckDB — dynamically queried, not dumped into context |
| Multi-agent pattern | Orchestrator → Collector → EDA → Hypothesis (4 agents, distinct system prompts, orchestrator-handoff) |
| Deployed | Cloud Run — see Live Demo above |
| README | This file |

---

## Grab-Bag Electives

### 1. Second data retrieval method (`app/tools/api_tool.py`)
Two distinct retrieval methods are used: SQL queries against the local DuckDB file for historical data, and live yfinance API calls for recent/current data. The Collector agent chooses between them based on the question's time scope. Implemented in `run_api_fetch` — supports `ohlcv`, `info`, and `earnings` fetch types.

### 2. Data visualisation (`app/tools/viz_tool.py`)
`run_viz` generates matplotlib charts at runtime — bar charts for categorical comparisons, line charts for time-series, scatter plots with regression lines for correlations. Charts are saved as PNG artifacts to `artifacts/` (persistent disk outputs) and returned as base64 for inline frontend rendering. Chart type is chosen dynamically based on the data shape.

### 3. Iterative refinement loop (`app/agents/orchestrator.py` → `run_pipeline` / `_run_pipeline_inner`)
After the EDA agent runs, the orchestrator checks `EDAFindings.sufficient`. If `False`, it re-invokes the Collector with `EDAFindings.refinement_hint` as a narrower query, then re-runs EDA on the new data. This repeats up to `MAX_REFINEMENTS=3` times. This implements the deep research pattern described in the requirements.

### 4. Structured output (`app/schemas.py`)
Pydantic schemas enforce structured output at every agent boundary:
- `RouteDecision` — intent classification output
- `CollectedData` — collector output handed to EDA
- `EDAFindings` — EDA output handed to hypothesis
- `HypothesisReport` — final hypothesis output
- `ChatResponse` — API response to frontend

All agent outputs are parsed and validated against these schemas before being passed downstream.

---

## Running Locally

### Prerequisites
- Python 3.13
- [uv](https://github.com/astral-sh/uv)
- Google Cloud project with Vertex AI enabled
- Authenticated: `gcloud auth application-default login`

### Setup

```bash
# Clone and install
git clone <repo-url>
cd project2
uv sync

# Set environment variables
export GOOGLE_CLOUD_PROJECT=my-project-4576-project2
export GOOGLE_CLOUD_REGION=us-central1   

# Build the database (one-time, ~5 minutes, requires internet)
python data/build_db.py

# Verify the database
python -c "
import duckdb
con = duckdb.connect('data/market.duckdb')
print(con.execute('SELECT MIN(date), MAX(date), COUNT(*) FROM ohlcv').fetchdf())
"

# Start the server
uvicorn app.main:app --reload --reload-dir app
```

Open `http://127.0.0.1:8000` in your browser.

**Diagnostic endpoint:** `http://127.0.0.1:8000/db-check` — confirms DB path and row counts.

---

## Project Structure

```
project2/
├── app/
│   ├── main.py              # FastAPI entry point, /chat /health /db-check
│   ├── schemas.py           # Pydantic structured output schemas
│   ├── agents/
│   │   ├── orchestrator.py  # Root orchestrator: routing, pipeline, refinement loop
│   │   ├── collector.py     # Collector sub-agent (ADK LlmAgent)
│   │   ├── eda.py           # EDA sub-agent (ADK LlmAgent)
│   │   └── hypothesis.py    # Hypothesis sub-agent (ADK LlmAgent)
│   └── tools/
│       ├── sql_tool.py      # Text2SQL + DuckDB executor
│       ├── api_tool.py      # Live yfinance API fetcher
│       ├── stats_tool.py    # Statistical aggregations (returns, volatility, EPS)
│       └── viz_tool.py      # Matplotlib chart generator → base64 PNG
├── data/
│   ├── build_db.py          # One-time DB builder (downloads from yfinance)
│   └── market.duckdb        # DuckDB database (generated, not committed to git)
├── eval/
│   ├── dataset.jsonl        # 25 test cases (in-domain, OOS, safety, edge)
│   └── run_eval.py          # Eval runner: deterministic checks + Claude MaaJ judge
├── static/
│   └── index.html           # Frontend UI
├── artifacts/               # Generated charts saved here at runtime
├── Dockerfile
├── pyproject.toml
├── cloudbuild.yaml
└── README.md
```

---

## Data Sources

| Source | Type | Contents | Size |
|---|---|---|---|
| yfinance (historical) | SQL / DuckDB | 5 years daily OHLCV, 50 tickers, 11 sectors | ~69k rows |
| yfinance (live) | REST API | Recent OHLCV, ticker info, earnings | Dynamic |
| yfinance (earnings) | REST API | Quarterly EPS actual vs estimate | 220 rows |

Tickers covered: AAPL, MSFT, NVDA, GOOGL, META, JNJ, UNH, PFE, ABBV, MRK, JPM, BAC, WFC, GS, MS, AMZN, TSLA, HD, MCD, NKE, PG, KO, PEP, WMT, COST, GE, CAT, HON, UPS, BA, XOM, CVX, COP, SLB, EOG, NEE, DUK, SO, AEP, EXC, PLD, AMT, EQIX, SPG, PSA, LIN, APD, SHW, FCX, NEM, VZ, T, NFLX, DIS, CMCSA