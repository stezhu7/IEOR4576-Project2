"""
schemas.py — Pydantic models used as structured-output contracts
between agents and as the API response shape.

Every agent boundary emits one of these; the orchestrator
validates them before passing to the next stage.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Output of the orchestrator's intent-classification step."""
    intent: Literal["in_scope", "oos", "safety"]
    reason: str = Field(description="One-sentence justification for the routing decision.")
    refined_query: str = Field(
        description="Cleaned, normalised version of the user query passed to the collector."
    )



class DataSource(BaseModel):
    source_type: Literal["sql", "api", "both"]
    sql_query: Optional[str]          = Field(None, description="The DuckDB SQL that was executed.")
    api_tickers: Optional[list[str]]  = Field(None, description="Tickers fetched live from yfinance.")
    row_count: int                    = Field(description="Total rows returned across all sources.")


class CollectedData(BaseModel):
    """Output of the Collector agent — handed to the EDA agent."""
    data_source: DataSource
    columns: list[str]
    preview: list[dict[str, Any]]     = Field(description="First 5 rows as list-of-dicts for context.")
    raw_json: str                     = Field(description="Full JSON-serialised DataFrame (compact).")
    collection_notes: str             = Field(description="What was collected and any caveats.")



class StatResult(BaseModel):
    metric: str
    value: Any
    unit: Optional[str] = None


class EDAFindings(BaseModel):
    """Output of the EDA agent — handed to the Hypothesis agent."""
    summary_stats: list[StatResult]   = Field(description="Key numeric findings.")
    anomalies: list[str]              = Field(description="Unusual patterns or outliers noticed.")
    chart_base64: Optional[str]       = Field(None, description="Base64-encoded PNG chart, if generated.")
    chart_title: Optional[str]        = None
    sufficient: bool                  = Field(
        description="True if findings are rich enough to form a hypothesis; "
                    "False triggers a refinement loop in the orchestrator."
    )
    refinement_hint: Optional[str]    = Field(
        None,
        description="If sufficient=False, a suggested narrower query for the collector."
    )
    eda_notes: str                    = Field(description="Narrative description of the exploration.")



class EvidencePoint(BaseModel):
    claim: str
    supporting_data: str              = Field(description="Specific numbers or patterns from EDA.")


class HypothesisReport(BaseModel):
    """Final output of the Hypothesis agent — returned to the user."""
    headline: str                     = Field(description="One-sentence summary of the hypothesis.")
    evidence: list[EvidencePoint]     = Field(description="Data-grounded evidence points.")
    caveats: list[str]                = Field(description="Limitations or alternative explanations.")
    chart_base64: Optional[str]       = Field(None, description="Visualisation to accompany the report.")
    chart_title: Optional[str]        = None
    confidence: Literal["high", "medium", "low"]
    full_narrative: str               = Field(description="Full analyst memo, 3-5 paragraphs.")



class ChatResponse(BaseModel):
    answer: str                       = Field(description="Human-readable response text.")
    backstop: str                     = Field(description="Routing tag for observability.")
    chart_base64: Optional[str]       = None
    chart_title: Optional[str]        = None
    eda_stats: Optional[list[dict]]   = None
    data_source: Optional[str]        = None
    confidence: Optional[str]         = None