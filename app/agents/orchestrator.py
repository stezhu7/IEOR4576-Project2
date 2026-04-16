"""
agents/orchestrator.py — Orchestrator Agent (root)

Responsibilities:
- Semantic intent classification (in_scope / oos / safety)
- Positive-framing OOS/safety responses
- Route in-scope queries through Collector → EDA → Hypothesis pipeline
- Drive the iterative refinement loop (up to MAX_REFINEMENTS)
- Assemble the final ChatResponse

SDK: google.genai (Vertex AI) — consistent across the entire codebase.
"""

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
You are a stock market data collector. You MUST call the run_text2sql tool to retrieve data.
Do NOT write SQL yourself. Do NOT guess table names. Do NOT return data you did not fetch.

The DuckDB database has exactly these tables:
- ohlcv        columns: date, ticker, sector, open, high, low, close, volume
- earnings     columns: ticker, sector, quarter, eps_estimate, eps_actual, surprise_percent
- sector_meta  columns: ticker, sector

CRITICAL RULE: Write SQL that returns AGGREGATED or SUMMARISED results (≤50 rows).
Never SELECT * or fetch thousands of raw rows. Always use GROUP BY, AVG, SUM, or LIMIT.

Examples of good queries for common questions:
- "highest sector return in 2023" — return ALL sectors ranked, not just top 1:
  WITH daily AS (
    SELECT ticker, sector, date, close,
      (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) /
      LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS dr
    FROM ohlcv
    WHERE date >= CAST('2023-01-01' AS DATE) AND date <= CAST('2023-12-31' AS DATE)
  )
  SELECT sector,
         ROUND(AVG(dr), 6)                        AS avg_daily_return,
         ROUND((POWER(1 + AVG(dr), 252) - 1), 4)  AS annualised_return,
         ROUND(STDDEV(dr) * SQRT(252), 4)          AS annualised_volatility,
         COUNT(*)                                  AS observations
  FROM daily WHERE dr IS NOT NULL
  GROUP BY sector ORDER BY annualised_return DESC

CRITICAL: NEVER use LIMIT 1 or LIMIT with any number when the question involves comparing
sectors or tickers. Always return ALL rows. A single row cannot be analysed or ranked.
The question asks which sector is HIGHEST — to answer that you must return ALL sectors ranked,
not just the top one. LIMIT 1 destroys the comparison entirely.

MANDATORY: Call run_text2sql(question=<the user's question>) FIRST.
Only after the tool returns results, build your JSON response.

Your response must be a JSON object with these exact fields:
{
  "data_source": {
    "source_type": "sql",
    "sql_query": "<copy the sql field from the tool result>",
    "api_tickers": null,
    "row_count": <copy row_count from tool result>
  },
  "columns": ["<copy columns list from tool result>"],
  "preview": [<first 3 items from preview in tool result>],
  "raw_json": "<copy json_data string from tool result — this must be the AGGREGATED result, ≤50 rows>",
  "collection_notes": "Retrieved N rows via SQL. Query aggregated by sector/ticker."
}

If the tool returns success=false or row_count=0, report that in collection_notes.
NEVER fabricate data or row counts.
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

STEP 2 — Call run_viz to generate a chart. Use these config rules:
  - If data has a "sector" column and a numeric column → use chart_type="bar",
    config={"title": "Sector Comparison", "xlabel": "Sector", "ylabel": "<metric name>",
            "label_col": "sector", "value_col": "<best numeric column>"}
  - If data has a "date" column → use chart_type="line"
  - If comparing two numeric columns → use chart_type="scatter"
  Pass json_data = the raw_json string (same as step 1)

STEP 3 — Return ONLY a JSON object with these exact fields:
{
  "summary_stats": [{"metric": "sector: <name> annualised_return", "value": 0.42, "unit": "ratio"}],
  "anomalies": ["..."],
  "chart_base64": "base64 PNG string from run_viz",
  "chart_title": "...",
  "sufficient": true,
  "refinement_hint": null,
  "eda_notes": "List every sector/ticker with its exact return value. Name the top and bottom performers."
}

CRITICAL rules:
- Pass raw_json (the array string) as json_data — NOT the whole input object
- summary_stats must have one entry PER sector/ticker, not just one overall entry
- eda_notes must name specific values: "Technology: 42.3%, Healthcare: 18.1%, ..."
- If row_count is 0, set sufficient=false and refinement_hint to suggest a broader query
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


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        ),
    )
    return (response.text or "").strip()


# ── Tool declaration builder ──────────────────────────────────────────────────

def _build_tools(funcs: list) -> list[types.Tool]:
    """
    Build google.genai Tool declarations from plain Python functions.
    Derives parameter schemas from type annotations via inspect.
    """
    declarations = []
    for fn in funcs:
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
                name=fn.__name__,
                description=doc,
                parameters={
                    "type": "OBJECT",
                    "properties": props,
                    "required": required,
                },
            )
        )

    return [types.Tool(function_declarations=declarations)]


# ── Agentic loop ──────────────────────────────────────────────────────────────

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

    # tool_config forces the model to call a tool on the first turn
    # rather than reasoning/hallucinating the answer directly
    forced_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.1,
        max_output_tokens=8192,
        tools=tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
    )
    # After first tool call, switch to AUTO so the model can choose to stop
    auto_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.1,
        max_output_tokens=8192,
        tools=tools,
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

        # Log finish reason if not STOP
        finish_reason = candidate.finish_reason if hasattr(candidate, "finish_reason") else "unknown"
        log.info("_agentic_loop turn %d: finish_reason=%s, parts=%d", turn, finish_reason, len(parts))

        fn_call_parts = [p for p in parts if p.function_call is not None]
        log.info("_agentic_loop turn %d: fn_calls=%s, text_parts=%d",
                 turn,
                 [p.function_call.name for p in fn_call_parts],
                 sum(1 for p in parts if p.text))

        if not fn_call_parts:
            # No tool calls — model has produced its final text answer
            text = "".join(p.text for p in parts if p.text).strip()
            log.info("_agentic_loop turn %d: final text length=%d, preview=%r", turn, len(text), text[:200])
            return text

        # Append model turn to history
        history.append(types.Content(role="model", parts=parts))

        # Execute each function call and collect responses
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
    return ""  # Exhausted max_turns without a text response


# ── Per-agent runners ─────────────────────────────────────────────────────────

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


def run_collector(question: str) -> CollectedData | None:
    from app.tools.sql_tool import run_text2sql
    from app.tools.api_tool import run_api_fetch

    log.info("run_collector: question=%r", question)
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


def run_eda(collected: CollectedData) -> EDAFindings | None:
    from app.tools.stats_tool import run_stats
    from app.tools.viz_tool   import run_viz

    log.info("run_eda: row_count=%d, columns=%s", collected.data_source.row_count, collected.columns)
    tool_map = {"run_stats": run_stats, "run_viz": run_viz}
    tools    = _build_tools([run_stats, run_viz])

    # Pass raw_json (the actual data rows) directly so run_stats receives
    # the row array, not a wrapper object.
    payload  = json.dumps({
        "raw_json":         collected.raw_json,
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
        return EDAFindings(**_parse_json(result))
    except Exception as exc:
        log.error("run_eda: JSON parse failed: %s", exc)
        return EDAFindings(
            summary_stats=[],
            anomalies=[],
            chart_base64=None,
            chart_title=None,
            sufficient=False,
            refinement_hint="EDA parse failed — retry with a simpler question.",
            eda_notes=f"Parse error: {exc}",
        )


def run_hypothesis(eda: EDAFindings) -> HypothesisReport | None:
    payload = json.dumps({
        "eda_findings": {
            "summary_stats": [s.model_dump() for s in eda.summary_stats],
            "anomalies":     eda.anomalies,
            "eda_notes":     eda.eda_notes,
        }
    })
    # Hypothesis needs no tools — pure reasoning over EDA structured output
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


# ── Main pipeline ─────────────────────────────────────────────────────────────

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

    # 1 — Collect
    collected = run_collector(query)
    if not collected:
        return ChatResponse(
            answer="I was unable to retrieve data for your question. Please try rephrasing.",
            backstop="collect_error",
        )

    # 2 — EDA + iterative refinement loop
    eda      = None
    attempts = 0
    for attempt in range(MAX_REFINEMENTS):
        attempts = attempt + 1
        eda = run_eda(collected)
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

    # 3 — Hypothesis
    hypothesis = run_hypothesis(eda)
    if not hypothesis:
        return ChatResponse(
            answer=eda.eda_notes,
            backstop="hypothesis_error",
            chart_base64=eda.chart_base64,
            chart_title=eda.chart_title,
        )

    # Assemble final answer
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