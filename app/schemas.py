"""Pydantic models: LLM structured outputs + public API contract."""

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Structured LLM outputs (used with `llm.with_structured_output(...)`)
# --------------------------------------------------------------------------- #
class QueryPlan(BaseModel):
    """How the agent will search for the user's question."""

    language: Literal["ja", "en"] = Field(description="Language of the user's question")
    standalone_question: str = Field(
        "",
        description="The current question rewritten so it can be understood without the chat history "
        "(resolve words like それ / that). Empty if the question is already standalone.",
    )
    search_queries: list[str] = Field(
        description="1-3 short JAPANESE keyword queries for searching Japanese white papers",
    )


class RelevanceGrade(BaseModel):
    relevant_ids: list[int] = Field(description="Numbers of the passages that help answer the question")


class DraftAnswer(BaseModel):
    answer: str = Field(description="Answer with inline citations like [1], [2]")


class GroundednessCheck(BaseModel):
    grounded: bool = Field(description="True only if every claim is supported by the passages")
    unsupported_claims: list[str] = Field(description="Claims not supported by the passages (empty if none)")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
Status = Literal["answered", "partially_supported", "insufficient_evidence"]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    doc_ids: list[str] | None = Field(default=None, description="Restrict search to these documents")
    thread_id: str | None = Field(
        default=None, max_length=64, description="Send the thread_id of the previous answer to ask a follow-up"
    )


class Citation(BaseModel):
    ref: int
    doc_id: str
    title: str
    ministry: str
    page: int
    snippet: str
    url: str  # opens the PDF at the cited page


class QualityCheck(BaseModel):
    """One row of the answer inspection sheet (see `graph.inspect_answer`)."""

    name: str
    passed: bool
    detail: str


class TraceStep(BaseModel):
    node: str
    label: str
    detail: str
    ms: int


class AskResponse(BaseModel):
    question: str
    standalone_question: str
    answer: str
    status: Status
    citations: list[Citation]
    checks: list[QualityCheck] = []
    unsupported_claims: list[str] = []
    numeric_issues: list[str] = []
    search_queries: list[str] = []
    trace: list[TraceStep] = []
    thread_id: str
    model: str
    total_tokens: int
    latency_ms: int
