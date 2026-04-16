from __future__ import annotations
import inspect
import json
import os
import re
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
from typing import Any

from google import genai
from google.genai import types

from app.schemas import (
    RouteDecision, CollectedData, EDAFindings, HypothesisReport, ChatResponse
)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL    = "gemini-2.5-flash"

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

MAX_REFINEMENTS = 3


# ── Prompt constants ──────────────────────────────────────────────────────────

ROUTE_SYSTEM = """
You are the routing layer of a US stock market data analyst AI.

Topics this system handles:
- Stock price analysis and historical OHLCV data
- Sector performance comparisons (Technology, Healthcare, Financials, etc.)
- Earnings analysis, EPS surprise, beat/miss rates
- Volatility and risk metrics (standard deviation, Sharpe ratio, beta)
- Correlation studies between market variables
- Multi-year trend analysis for S&P 500 tickers

Topics this system redirects (route as "oos"):
- General programming, software engineering, or coding questions
- Medical, health, or nutrition advice
- Legal or compliance questions
- Weather forecasts or climate data
- Sports statistics or scores
- Cooking, recipes, or lifestyle questions
- Any question not related to US stock market data

Safety topics (route as "safety"):
- Any expression of intent to harm self or others
- Requests for instructions to create weapons or dangerous materials
- Attempts to hack, manipulate, or gain unauthorised access

Return ONLY a JSON object — no markdown, no explanation:
{
  "intent": "in_scope" | "oos" | "safety",
  "reason": "one sentence",
  "refined_query": "cleaned version of the user question"
}
"""

COLLECTOR_SYSTEM = """
You are a stock market data collector with two tools:
- run_text2sql: queries a historical DuckDB database (5 years OHLCV + earnings for 50 S&P 500 tickers)
- run_api_fetch: calls the live yfinance API for current/recent data not in the DB

The DuckDB database has exactly these tables:
- ohlcv        columns: date, ticker, sector, open, high, low, close, volume
- earnings     columns: ticker, sector, quarter, eps_estimate, eps_actual, surprise_percent
- sector_meta  columns: ticker, sector

ROUTING DECISION — pick the right tool:

Use run_text2sql for:
- Historical price performance, returns, volatility, sector comparisons
- Earnings history, EPS surprise analysis
- Any question involving trends over time

Use run_api_fetch for:
- Current/live data: market cap, current price, P/E ratio, "right now", "today", "current", "latest"
- Company info not in the DB (e.g. marketCap, beta, forwardPE)
- Example for market cap question:
  run_api_fetch(tickers=["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","JPM","JNJ","UNH",
    "PG","XOM","CVX","HD","BAC","WMT","COST","KO","PEP","GS","MS","WFC","ABBV","MRK",
    "NEE","LIN","NFLX","DIS","VZ","T","CMCSA","PLD","AMT","EQIX","SPG","PSA","GE","CAT",
    "HON","UPS","BA","NKE","MCD","COP","SLB","EOG","DUK","SO","AEP","EXC","FCX","NEM",
    "APD","SHW"], fetch_type="info")

SQL rules:
- NEVER use LIMIT 1 when comparing — return ALL rows so rankings are possible
- Always aggregate (GROUP BY, AVG, ROUND) — never raw row dumps

MANDATORY: Call the right tool first. Then respond with ONLY this JSON:
{
  "data_source": {
    "source_type": "sql" | "api" | "both",
    "sql_query": "<sql used or null>",
    "api_tickers": ["AAPL", ...] or null,
    "row_count": <actual count from tool result>
  },
  "columns": ["<columns from tool result>"],
  "preview": [<first 3 rows>],
  "raw_json": "<json_data string from tool result>",
  "collection_notes": "Retrieved N rows via <method>. <brief description>"
}

NEVER fabricate data. If a tool returns 0 rows, report it in collection_notes.
"""

EDA_SYSTEM = """
You are a stock market EDA analyst.

You receive a JSON object with these fields:
- raw_json: a JSON array string of data rows (this is what you analyse)
- columns: list of column names
- collection_notes: description of what was collected
- row_count: number of rows

STEP 1 — Call run_stats:
  json_data = the raw_json string (the array of rows)
  analysis_type = "auto"

STEP 2 — Call run_viz to generate a chart. Use flat parameters (NOT a config dict):
  - Sector comparison → chart_type="bar", label_col="sector",
    value_col="<best numeric column e.g. annualised_return>",
    title="Sector Returns", xlabel="Sector", ylabel="Annualised Return"
  - Ticker/company ranking (e.g. market cap) → chart_type="bar", label_col="ticker",
    value_col="marketCap", title="Top Companies by Market Cap",
    xlabel="Ticker", ylabel="Market Cap (USD)", top_n="10"
  - Time series → chart_type="line", x_col="date", y_cols="close"
  - Correlation → chart_type="scatter", x_col="<col1>", y_col="<col2>", label_col="ticker"
  Always pass json_data = the raw_json string

STEP 3 — Return ONLY a JSON object with these exact fields:
{
  "summary_stats": [{"metric": "<ticker/sector> <metric_name>", "value": 0.42, "unit": "..."}],
  "anomalies": ["..."],
  "chart_base64": null,
  "chart_title": "...",
  "sufficient": true,
  "refinement_hint": null,
  "eda_notes": "List every item with its exact value. Name top and bottom performers."
}

CRITICAL rules:
- Pass raw_json (the array string) as json_data to BOTH run_stats and run_viz — NOT the wrapper
- summary_stats: one entry PER sector/ticker with the key metric value
- eda_notes: cite specific numbers e.g. "AAPL marketCap=$3.2T, MSFT=$2.9T..."
- Always set chart_base64=null in your JSON — the chart is injected automatically
- chart_title: describe what the chart shows
- For market cap questions: use label_col="ticker", value_col="marketCap", sort by marketCap DESC
- If row_count is 0: set sufficient=false, add refinement_hint
"""

HYPOTHESIS_SYSTEM = """
You are a stock market analyst writing a concise data-driven memo.

Given EDA findings (as JSON), form a clear hypothesis with ranked comparisons.

Rules:
- Cite SPECIFIC numbers from summary_stats (e.g. "Technology: 42.3% annualised return")
- When multiple sectors/tickers are present, RANK them explicitly (1st, 2nd, 3rd...)
- Name the winner AND the worst performer — contrast makes the hypothesis stronger
- Keep full_narrative to 2-3 focused paragraphs, not 5 vague ones
- Do NOT say "further analysis required" — draw a conclusion from what you have
- Do NOT use general market knowledge — only numbers from the EDA input

Return ONLY a JSON object with these exact fields:
{
  "headline": "one-sentence hypothesis naming the top performer and its return",
  "evidence": [{"claim": "...", "supporting_data": "exact metric=value from EDA"}],
  "caveats": ["..."],
  "chart_base64": null,
  "chart_title": null,
  "confidence": "high" | "medium" | "low",
  "full_narrative": "2-3 paragraphs with specific numbers, rankings, and a clear conclusion"
}
"""

OOS_RESPONSE = (
    "This assistant specialises in US stock market analysis — "
    "including price trends, sector comparisons, earnings analysis, and volatility metrics.\n\n"
    "Your question appears to be outside that scope. "
    "Topics this system redirects include: general coding, medical advice, legal questions, "
    "weather, sports, and other non-market questions.\n\n"
    "Try asking something like: *\"Which S&P 500 sector had the highest return in 2023?\"* "
    "or *\"Is there a correlation between earnings surprise and next-day price movement?\"*"
)

SAFETY_RESPONSE = (
    "I'm not able to help with that request.\n\n"
    "If you or someone you know is in danger, please contact emergency services (US: 911) "
    "or the Crisis & Suicide Lifeline (call/text 988 in the US).\n\n"
    "I'm here to help with US stock market analysis whenever you're ready."
)



def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from LLM output."""
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def _gemini_call(system: str, user: str, temperature: float = 0.0) -> str:
    """Single-turn Gemini call via google.genai — no tools."""
    response = _client.models.generate_content(
        model=MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()



def _build_tools(funcs: list) -> list[types.Tool]:
    declarations = []
    for entry in funcs:
        # entry can be (fn, "name") to override the tool name exposed to the model
        if isinstance(entry, tuple):
            fn, tool_name = entry
        else:
            fn, tool_name = entry, entry.__name__
        sig      = inspect.signature(fn)
        doc      = (fn.__doc__ or fn.__name__).strip().split("\n")[0]
        props: dict[str, Any] = {}
        required: list[str]   = []

        for pname, param in sig.parameters.items():
            ann = param.annotation
            if ann in (str, inspect.Parameter.empty):
                props[pname] = {"type": "STRING"}
            elif ann == list:
                props[pname] = {"type": "ARRAY", "items": {"type": "STRING"}}
            elif ann == dict:
                props[pname] = {"type": "OBJECT"}
            elif ann == int:
                props[pname] = {"type": "INTEGER"}
            elif ann == float:
                props[pname] = {"type": "NUMBER"}
            elif ann == bool:
                props[pname] = {"type": "BOOLEAN"}
            else:
                props[pname] = {"type": "STRING"}

            if param.default is inspect.Parameter.empty:
                required.append(pname)

        declarations.append(
            types.FunctionDeclaration(
                name=tool_name,
                description=doc,
                parameters={
                    "type": "OBJECT",
                    "properties": props,
                    "required": required,
                },
            )
        )

    return [types.Tool(function_declarations=declarations)]



def _agentic_loop(
    system: str,
    user_input: str,
    tool_map: dict[str, Any],
    tools: list[types.Tool],
    max_turns: int = 6,
) -> str:
    """
    Multi-turn agentic loop using google.genai exclusively.
    Executes tool calls until the model returns a plain text response.
    """
    history: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_input)])
    ]

    thinking_cfg = types.ThinkingConfig(thinking_budget=0)

    forced_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.1,
        max_output_tokens=8192,
        tools=tools,
        thinking_config=thinking_cfg,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
    )
    auto_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.1,
        max_output_tokens=8192,
        tools=tools,
        thinking_config=thinking_cfg,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        ),
    )

    for turn in range(max_turns):
        active_config = forced_config if turn == 0 and tools else auto_config
        try:
            response = _client.models.generate_content(
                model=MODEL,
                contents=history,
                config=active_config,
            )
        except Exception as exc:
            log.error("_agentic_loop: generate_content failed on turn %d: %s", turn, exc)
            return ""

        candidate = response.candidates[0]
        parts     = candidate.content.parts

        finish_reason = candidate.finish_reason if hasattr(candidate, "finish_reason") else "unknown"
        log.info("_agentic_loop turn %d: finish_reason=%s, parts=%d", turn, finish_reason, len(parts))

        fn_call_parts = [p for p in parts
                         if p.function_call is not None
                         and not getattr(p, "thought", False)]
        text_parts    = [p for p in parts
                         if p.text
                         and not getattr(p, "thought", False)]

        log.info("_agentic_loop turn %d: fn_calls=%s, text_parts=%d",
                 turn,
                 [p.function_call.name for p in fn_call_parts],
                 len(text_parts))

        if not fn_call_parts:
            text = "".join(p.text for p in text_parts).strip()
            log.info("_agentic_loop turn %d: final text length=%d, preview=%r", turn, len(text), text[:200])
            return text

        history.append(types.Content(role="model", parts=parts))

        tool_response_parts: list[types.Part] = []
        for part in fn_call_parts:
            fn_name = part.function_call.name
            fn_args = dict(part.function_call.args) if part.function_call.args else {}
            fn_fn   = tool_map.get(fn_name)

            if fn_fn:
                try:
                    result = fn_fn(**fn_args)
                except Exception as exc:
                    log.error("Tool %s failed: %s", fn_name, exc)
                    result = {"error": str(exc)}  # inside except block
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_name,
                        response={"result": result},
                    )
                )
            )

        history.append(types.Content(role="tool", parts=tool_response_parts))

    log.warning("_agentic_loop: exhausted %d turns without text response", max_turns)
    return ""  



def classify_intent(user_text: str) -> RouteDecision:
    """Semantic intent classification — single call, no tools."""
    raw = _gemini_call(ROUTE_SYSTEM, user_text, temperature=0.0)
    try:
        return RouteDecision(**_parse_json(raw))
    except Exception as exc:
        log.warning("classify_intent parse failed: %s | raw=%r", exc, raw)
        return RouteDecision(
            intent="in_scope",
            reason="Classification parse failed, defaulting to in_scope.",
            refined_query=user_text,
        )


ALL_TICKERS = [
    "AAPL","MSFT","NVDA","GOOGL","META","JNJ","UNH","PFE","ABBV","MRK",
    "JPM","BAC","WFC","GS","MS","AMZN","TSLA","HD","MCD","NKE",
    "PG","KO","PEP","WMT","COST","GE","CAT","HON","UPS","BA",
    "XOM","CVX","COP","SLB","EOG","NEE","DUK","SO","AEP","EXC",
    "PLD","AMT","EQIX","SPG","PSA","LIN","APD","SHW","FCX","NEM",
    "VZ","T","NFLX","DIS","CMCSA",
]


def _try_sql_then_api(question: str) -> CollectedData | None:
    from app.tools.sql_tool import run_text2sql
    from app.tools.api_tool import run_api_fetch

    sql_result = run_text2sql(question)
    log.info("_try_sql_then_api: sql needs_api=%s, row_count=%d",
             sql_result.get("needs_api"), sql_result.get("row_count", 0))

    if sql_result.get("needs_api") or not sql_result.get("success"):
        log.info("_try_sql_then_api: falling back to run_api_fetch(info)")
        api_result = run_api_fetch(tickers=ALL_TICKERS, fetch_type="info")

        if not api_result.get("success", True):
            return None

        rows = []
        for ticker, info in api_result.items():
            if isinstance(info, dict) and "error" not in info:
                info["ticker"] = ticker
                rows.append(info)

        if not rows:
            return None

        import pandas as pd
        df = pd.DataFrame(rows)
        # Keep only useful columns to avoid huge payloads
        keep = ["ticker","shortName","sector","marketCap","trailingPE","forwardPE",
                "currentPrice","beta","dividendYield","earningsGrowth","revenueGrowth",
                "fiftyTwoWeekHigh","fiftyTwoWeekLow","targetMeanPrice"]
        df = df[[c for c in keep if c in df.columns]]
        df = df.dropna(subset=["marketCap"] if "marketCap" in df.columns else [])

        json_data = df.to_json(orient="records")
        return CollectedData(
            data_source={
                "source_type": "api",
                "sql_query":   None,
                "api_tickers": list(df["ticker"]) if "ticker" in df.columns else ALL_TICKERS,
                "row_count":   len(df),
            },
            columns=list(df.columns),
            preview=df.head(3).to_dict(orient="records"),
            raw_json=json_data,
            collection_notes=f"Fetched live info for {len(df)} tickers via yfinance API.",
        )

    import pandas as pd, json as _json
    return CollectedData(
        data_source={
            "source_type": "sql",
            "sql_query":   sql_result["sql"],
            "api_tickers": None,
            "row_count":   sql_result["row_count"],
        },
        columns=sql_result["columns"],
        preview=sql_result["preview"],
        raw_json=sql_result["json_data"],
        collection_notes=f"Retrieved {sql_result['row_count']} rows via SQL.",
    )


def run_collector(question: str) -> CollectedData | None:
    from app.tools.sql_tool import run_text2sql
    from app.tools.api_tool import run_api_fetch

    log.info("run_collector: question=%r", question)

    direct = _try_sql_then_api(question)
    if direct is not None:
        log.info("run_collector: direct result row_count=%d source=%s",
                 direct.data_source.row_count, direct.data_source.source_type)
        return direct

    log.info("run_collector: falling back to LLM agentic loop")
    tool_map = {"run_text2sql": run_text2sql, "run_api_fetch": run_api_fetch}
    tools    = _build_tools([run_text2sql, run_api_fetch])
    result   = _agentic_loop(COLLECTOR_SYSTEM, question, tool_map, tools)
    log.info("run_collector: raw result=%r", result[:300] if result else None)

    if not result:
        log.error("run_collector: _agentic_loop returned empty string")
        return None
    try:
        return CollectedData(**_parse_json(result))
    except Exception as exc:
        log.error("run_collector: JSON parse failed: %s | raw=%r", exc, result[:300])
        return CollectedData(
            data_source={"source_type": "sql", "sql_query": None,
                         "api_tickers": None, "row_count": 0},
            columns=[],
            preview=[],
            raw_json="[]",
            collection_notes=f"Parse error: {exc}. Raw: {result[:300]}",
        )


def run_eda(collected: CollectedData, original_question: str = "") -> EDAFindings | None:
    from app.tools.stats_tool import run_stats
    from app.tools.viz_tool   import run_viz

    log.info("run_eda: row_count=%d, columns=%s", collected.data_source.row_count, collected.columns)

    captured_chart: dict = {}

    def run_viz_capturing(
        json_data: str, chart_type: str,
        title: str = "Chart", xlabel: str = "", ylabel: str = "",
        label_col: str = "", value_col: str = "",
        x_col: str = "", y_col: str = "", y_cols: str = "",
        top_n: str = ""
    ) -> dict:
        result = run_viz(
            json_data=json_data, chart_type=chart_type,
            title=title, xlabel=xlabel, ylabel=ylabel,
            label_col=label_col, value_col=value_col,
            x_col=x_col, y_col=y_col, y_cols=y_cols,
            top_n=top_n,
        )
        if result.get("success") and result.get("chart_base64"):
            captured_chart["b64"]   = result["chart_base64"]
            captured_chart["title"] = result.get("chart_title", "")
            log.info("run_eda: chart captured, b64 length=%d", len(captured_chart["b64"]))
        else:
            log.warning("run_eda: run_viz returned no chart: %s", result.get("error"))
        return result

    tool_map = {"run_stats": run_stats, "run_viz": run_viz_capturing}
    tools    = _build_tools([run_stats, (run_viz_capturing, "run_viz")])

    raw_json = collected.raw_json
    if collected.data_source.source_type == "api" and len(raw_json) > 3000:
        try:
            import json as _j, pandas as _pd
            _df = _pd.DataFrame(_j.loads(raw_json))
            if "marketCap" in _df.columns:
                _df = _df.sort_values("marketCap", ascending=False).head(20)
            else:
                _df = _df.head(20)
            raw_json = _df.to_json(orient="records")
            log.info("run_eda: trimmed API payload to %d rows, %d chars",
                     len(_df), len(raw_json))
        except Exception as e:
            log.warning("run_eda: payload trimming failed: %s", e)

    payload = json.dumps({
        "raw_json":         raw_json,
        "columns":          collected.columns,
        "collection_notes": collected.collection_notes,
        "row_count":        collected.data_source.row_count,
    })
    result = _agentic_loop(EDA_SYSTEM, payload, tool_map, tools)
    log.info("run_eda: raw result=%r", result[:300] if result else None)

    if not result:
        log.error("run_eda: _agentic_loop returned empty")
        return None
    try:
        findings = EDAFindings(**_parse_json(result))
        # Inject captured chart if the LLM forgot to include it
        if captured_chart and not findings.chart_base64:
            log.info("run_eda: injecting captured chart into findings")
            findings.chart_base64 = captured_chart.get("b64")
            findings.chart_title  = captured_chart.get("title")
        return findings
    except Exception as exc:
        log.error("run_eda: JSON parse failed: %s", exc)
        eda = EDAFindings(
            summary_stats=[],
            anomalies=[],
            chart_base64=captured_chart.get("b64"),
            chart_title=captured_chart.get("title"),
            sufficient=False,
            refinement_hint=original_question or "Retry with aggregated sector or ticker data.",
            eda_notes=f"Parse error: {exc}",
        )
        return eda


def run_hypothesis(eda: EDAFindings) -> HypothesisReport | None:
    payload = json.dumps({
        "eda_findings": {
            "summary_stats": [s.model_dump() for s in eda.summary_stats],
            "anomalies":     eda.anomalies,
            "eda_notes":     eda.eda_notes,
        }
    })
    result = _gemini_call(HYPOTHESIS_SYSTEM, payload, temperature=0.2)

    if not result:
        return None
    try:
        return HypothesisReport(**_parse_json(result))
    except Exception as exc:
        return HypothesisReport(
            headline="Unable to form hypothesis.",
            evidence=[],
            caveats=[str(exc)],
            confidence="low",
            full_narrative=result[:500] if result else "No output.",
        )



def run_pipeline(user_text: str) -> ChatResponse:
    """
    Full orchestrator pipeline:
      0. Semantic intent classification  (google.genai, no tools)
      1. Collect  — SQL + optional API   (google.genai agentic loop)
      2. EDA      — stats + viz          (google.genai agentic loop, refinement loop)
      3. Hypothesize — grounded narrative (google.genai, no tools)
    """

    log.info("run_pipeline: user_text=%r", user_text)
    try:
        return _run_pipeline_inner(user_text)
    except Exception as exc:
        log.exception("run_pipeline: unhandled exception")
        return ChatResponse(
            answer=f"An unexpected error occurred: {exc}",
            backstop=f"exception:{type(exc).__name__}",
        )


def _run_pipeline_inner(user_text: str) -> ChatResponse:
    # 0 — Route
    route = classify_intent(user_text)

    if route.intent == "safety":
        return ChatResponse(answer=SAFETY_RESPONSE, backstop="safety")
    if route.intent == "oos":
        return ChatResponse(answer=OOS_RESPONSE, backstop="oos")

    query = route.refined_query

    collected = run_collector(query)
    if not collected:
        return ChatResponse(
            answer="I was unable to retrieve data for your question. Please try rephrasing.",
            backstop="collect_error",
        )

    eda      = None
    attempts = 0
    for attempt in range(MAX_REFINEMENTS):
        attempts = attempt + 1
        eda = run_eda(collected, original_question=query)
        if not eda:
            break
        if eda.sufficient:
            break
        if eda.refinement_hint:
            refined = run_collector(eda.refinement_hint)
            if refined and refined.data_source.row_count > 0:
                collected = refined
                continue
        break

    if not eda:
        return ChatResponse(
            answer="Data was collected but analysis failed. Please try again.",
            backstop="eda_error",
        )

    hypothesis = run_hypothesis(eda)
    if not hypothesis:
        return ChatResponse(
            answer=eda.eda_notes,
            backstop="hypothesis_error",
            chart_base64=eda.chart_base64,
            chart_title=eda.chart_title,
        )

    lines = [f"## {hypothesis.headline}\n", hypothesis.full_narrative, "\n\n**Evidence:**"]
    for ev in hypothesis.evidence:
        lines.append(f"- **{ev.claim}**: {ev.supporting_data}")
    if hypothesis.caveats:
        lines.append("\n**Caveats:**")
        for c in hypothesis.caveats:
            lines.append(f"- {c}")
    lines.append(f"\n*Confidence: {hypothesis.confidence}*")

    return ChatResponse(
        answer="\n".join(lines),
        backstop=f"pipeline/ok/attempts={attempts}",
        chart_base64=eda.chart_base64,
        chart_title=eda.chart_title,
        eda_stats=[s.model_dump() for s in eda.summary_stats],
        data_source=collected.data_source.source_type,
        confidence=hypothesis.confidence,
    )