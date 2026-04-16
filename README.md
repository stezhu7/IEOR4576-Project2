# Stock Market Analyst AI

A multi-agent system that performs the first three steps of a data analysis lifecycle — **Collect → EDA → Hypothesize** — over real US stock market data.

Ask questions like:
- *"Which S&P 500 sector had the highest average return in 2023?"*
- *"Compare XXX volatility over the last 2 years"*
- *"What company has the highest market cap in S&P 500 right now?"*

---

## Live Demo

**Deployed URL:** `https://ieor4576-project2-git-7610618360.us-central1.run.app/`

---

## The Three Steps

### Step 1 — Collect (`app/agents/orchestrator.py` → `run_collector`, `_try_sql_then_api`)

The Collector retrieves real data at runtime using a **two-stage routing system** (`_try_sql_then_api`):

**Stage A — Schema-aware routing (`app/tools/sql_tool.py` → `run_text2sql`)**

The system first attempts to answer via SQL. `generate_sql` translates the user's question into a DuckDB query using Gemini. If the question asks for data not in the DB (market cap, P/E ratio, current price, beta), `generate_sql` returns the sentinel `NEEDS_API` instead of SQL. A secondary field scanner (`_references_api_only_fields`) catches cases where the model ignores the sentinel. This makes routing **deterministic** — no LLM guess required.

The DuckDB database (`data/market.duckdb`) contains:
- `ohlcv` — ~69,000 rows of daily OHLCV prices for 50 S&P 500 tickers across 11 GICS sectors, 5 years
- `earnings` — 220 rows of quarterly EPS actual vs estimate with surprise percentages
- `sector_meta` — ticker-to-sector mapping

The dataset is too large to load into context — SQL is generated dynamically per question.

**Stage B — Live yfinance API (`app/tools/api_tool.py` → `run_api_fetch`)**

When Stage A signals `NEEDS_API` or returns 0 rows, the collector automatically falls back to `run_api_fetch` with `fetch_type="info"` to retrieve live market data (market cap, P/E, beta, current price, etc.) for all 50 tickers.

---

### Step 2 — EDA (`app/agents/orchestrator.py` → `run_eda`)

The EDA agent performs exploratory data analysis using two tool calls before forming any conclusion:

**Tool 1 — Statistical aggregation (`app/tools/stats_tool.py` → `run_stats`)**

Auto-detects data type and applies the appropriate computation:
- Raw OHLCV data → `compute_return_stats`, `compute_sector_stats`, `detect_anomalies`
- Pre-aggregated results (sector returns, ticker metrics) → `compute_precomputed_stats`
- Market info data (marketCap, PE, beta) → `compute_precomputed_stats`

Returns per-sector or per-ticker stats with specific numeric values.

**Tool 2 — Data visualisation (`app/tools/viz_tool.py` → `run_viz`)**

Generates a matplotlib chart (bar for rankings/comparisons, line for time-series, scatter for correlations), saves as PNG artifact to `artifacts/`, and returns base64 for inline frontend rendering. Chart type and axis mapping are determined by the data shape. The chart is captured directly in Python (`run_viz_capturing` wrapper) to ensure it reaches the response even if the LLM omits it from its JSON output.

The EDA is dynamic — sector returns produce a ranked bar chart; volatility questions produce a per-ticker bar chart; correlation questions produce a scatter plot with regression line.

---

### Step 3 — Hypothesize (`app/agents/orchestrator.py` → `run_hypothesis`)

The Hypothesis agent receives the EDA findings and writes a data-grounded analyst memo:
- Names the top and bottom performers with exact figures from EDA
- Lists evidence points each citing a specific metric value
- Ranks all items explicitly (1st, 2nd, 3rd...)
- Acknowledges caveats and assigns a confidence level (high / medium / low)

The hypothesis is derived exclusively from `EDAFindings.summary_stats` — the system prompt explicitly prohibits using general market knowledge from model weights.

---

## Architecture

```
User query
    │
    ▼
Orchestrator — semantic intent classification (ROUTE_SYSTEM)
    │
    ├── OOS  → redirect message (topics outside stock market analysis)
    ├── Safety → crisis response
    │
    └── In-scope
            │
            ▼
        _try_sql_then_api (deterministic routing)
        ├── run_text2sql → DuckDB SQL  (historical data)
        │     └── NEEDS_API sentinel → automatic fallback
        └── run_api_fetch → yfinance   (live/fundamental data)
            │
            ▼
        run_eda  ◄──── iterative refinement loop (max 3×)
        ├── run_stats   (aggregations, anomaly detection)
        └── run_viz     (bar / line / scatter → PNG artifact)
            │
            ▼
        run_hypothesis
        └── grounded narrative + evidence + caveats
            │
            ▼
        Frontend
```

**Multi-agent pattern:** Four distinct agents each defined by a separate system prompt in `app/agents/orchestrator.py`:
- `ROUTE_SYSTEM` — intent classifier (in_scope / oos / safety)
- `COLLECTOR_SYSTEM` — data retrieval agent with routing rules
- `EDA_SYSTEM` — exploration agent with tool-call instructions
- `HYPOTHESIS_SYSTEM` — analyst memo writer

The agents communicate via structured Pydantic schemas (`app/schemas.py`) and are orchestrated by `_run_pipeline_inner`. The agentic loop (`_agentic_loop`) handles multi-turn tool calling for the Collector and EDA agents using the `google-genai` Vertex AI SDK directly.

---

## Core Requirements

| Requirement | Implementation |
|---|---|
| Frontend | `static/index.html` — dark-themed chat UI with inline chart rendering, stat pills, pipeline progress indicator, source badge (SQL/API) |
| Agent framework | `google-genai` (Vertex AI) — custom multi-turn agentic loop in `app/agents/orchestrator.py` → `_agentic_loop`, with 4 agents each defined by a distinct system prompt |
| Tool calling | `run_text2sql`, `run_api_fetch`, `run_stats`, `run_viz` — all called at runtime via the agentic loop |
| Non-trivial dataset | 69,025 OHLCV rows + 220 earnings rows in DuckDB — dynamically queried, never dumped into context |
| Multi-agent pattern | Orchestrator-handoff: 4 agents with distinct system prompts and responsibilities, driven by `_run_pipeline_inner` |
| Deployed | Cloud Run — see Live Demo above |
| README | This file |

---

## Grab-Bag Electives

### 1. Second data retrieval method (`app/tools/api_tool.py` → `run_api_fetch`)

Two distinct retrieval methods: SQL queries against DuckDB for historical price/earnings data, and live yfinance API calls for current fundamentals (market cap, P/E, beta, current price). Routing is deterministic via the `NEEDS_API` sentinel in `sql_tool.py` — if the generated SQL references fields not in the DB schema (e.g. `marketcap`, `trailingpe`), the system automatically falls back to the API without LLM involvement.

### 2. Data visualisation (`app/tools/viz_tool.py` → `run_viz`, `bar_chart`, `line_chart`, `scatter_chart`)

`run_viz` generates matplotlib charts at runtime with flat string parameters (no opaque dict). Chart type is chosen by the EDA agent based on data shape: bar for categorical rankings, line for time-series, scatter with regression line for correlations. Charts are saved as timestamped PNG artifacts to `artifacts/` and returned as base64 for inline frontend rendering. A `run_viz_capturing` wrapper in `run_eda` ensures the chart reaches the response even if the LLM omits it.

### 3. Iterative refinement loop (`app/agents/orchestrator.py` → `_run_pipeline_inner`)

After the EDA agent runs, the orchestrator checks `EDAFindings.sufficient`. If `False`, it re-invokes the collector with the original user question (not a generic hint) and re-runs EDA on the refreshed data. This repeats up to `MAX_REFINEMENTS=3` times. Implements the deep research pattern.

### 4. Structured output (`app/schemas.py`)

Pydantic schemas enforce structured output at every agent boundary:
- `RouteDecision` — intent classification (intent, reason, refined_query)
- `CollectedData` — collector output (data_source, columns, raw_json, preview)
- `EDAFindings` — EDA output (summary_stats, anomalies, chart_base64, sufficient, refinement_hint)
- `HypothesisReport` — final output (headline, evidence, caveats, confidence, full_narrative)
- `ChatResponse` — API response to frontend

---

## Running Locally

### Prerequisites
- Python 3.13
- [uv](https://github.com/astral-sh/uv)
- Google Cloud project with Vertex AI API enabled
- Authenticated: `gcloud auth application-default login`

### Setup

```bash
# Clone and install
git clone <repo-url>
cd project2
uv sync

# Set environment variables
export GOOGLE_CLOUD_PROJECT=my-project-4576-project2
export GOOGLE_CLOUD_REGION=us-central1   # optional, defaults to us-central1

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
│   ├── main.py              # FastAPI entry: /chat /health /db-check
│   ├── schemas.py           # Pydantic structured output schemas
│   ├── agents/
│   │   └── orchestrator.py  # All pipeline logic: 4 agent system prompts,
│   │                        # _agentic_loop, _try_sql_then_api, run_collector,
│   │                        # run_eda, run_hypothesis, run_pipeline
│   └── tools/
│       ├── sql_tool.py      # Text2SQL + DuckDB executor + NEEDS_API routing
│       ├── api_tool.py      # Live yfinance API fetcher
│       ├── stats_tool.py    # Statistical aggregations (returns, volatility, EPS)
│       └── viz_tool.py      # Matplotlib chart generator → base64 PNG artifacts
├── data/
│   ├── build_db.py          # One-time DB builder (downloads from yfinance)
│   └── market.duckdb        # DuckDB database (generated, not committed to git)
├── eval/
│   ├── dataset.jsonl        # 25 test cases (in-domain, OOS, safety, edge)
│   └── run_eval.py          # Eval runner: deterministic + Claude MaaJ judge
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
| yfinance (live) | REST API | Current price, market cap, P/E, beta, etc. | 50 tickers |
| yfinance (earnings) | REST API / DB | Quarterly EPS actual vs estimate | 220 rows |

**Tickers:** AAPL MSFT NVDA GOOGL META JNJ UNH PFE ABBV MRK JPM BAC WFC GS MS AMZN TSLA HD MCD NKE PG KO PEP WMT COST GE CAT HON UPS BA XOM CVX COP SLB EOG NEE DUK SO AEP EXC PLD AMT EQIX SPG PSA LIN APD SHW FCX NEM VZ T NFLX DIS CMCSA

**Sectors:** Technology, Healthcare, Financials, Consumer Discretionary, Consumer Staples, Industrials, Energy, Utilities, Real Estate, Materials, Communication Services