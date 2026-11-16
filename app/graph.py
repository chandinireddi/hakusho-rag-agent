"""Corrective RAG agent built with LangGraph.

    analyze ─► retrieve ─► grade ─┬─(relevant found)──► generate ─► verify ─► END
                  ▲               ├─(none, retries left)► rewrite ─┘(back to retrieve)
                  └───────────────┘
                                  └─(none, no retries)─► no_answer ─► END

Why an agent instead of a single prompt?
* analyze    : resolves follow-up questions ("それはいつ?") using the conversation history
* grade      : drops passages that only *look* relevant (precision before generation)
* rewrite    : self-corrects a failed search instead of answering from weak context
* verify     : 4-item answer inspection (citations present / no invented citation numbers /
               numbers match the cited pages / LLM fact-check). "answered" only if all 4 pass.
* no_answer  : the system says "not found" rather than hallucinating

Conversation memory uses a LangGraph checkpointer: the state of each `thread_id` is kept
between requests, and only the `history` field accumulates across turns.
"""

import operator
import re
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app import prompts
from app.config import Settings
from app.grounding import check_numbers
from app.ingest import Chunk
from app.retriever import Hit, HybridRetriever
from app.schemas import DraftAnswer, GroundednessCheck, QueryPlan, RelevanceGrade

NODE_LABELS = {
    "analyze": "質問を解析",
    "retrieve": "ハイブリッド検索",
    "grade": "関連性を評価",
    "rewrite": "検索クエリを再生成",
    "generate": "根拠付き回答を生成",
    "verify": "回答を検品（4項目）",
    "no_answer": "回答不可と判定",
}


class AgentState(TypedDict, total=False):
    # ---- per-turn input (overwritten by every request) ----
    question: str
    doc_ids: list[str] | None
    # ---- per-turn working state ----
    language: str
    standalone_question: str
    search_queries: list[str]
    tried_queries: list[str]
    rewrites: int
    candidates: list[Hit]
    context: list[Chunk]
    answer: str
    cited_ids: list[int]
    status: str
    unsupported_claims: list[str]
    numeric_issues: list[str]
    removed_citations: int  # citation numbers the LLM invented (removed by renumber_citations)
    checks: list[dict]  # answer inspection results: {"name", "passed", "detail"}
    trace: list[dict]  # no reducer on purpose: the request sends trace=[] to reset it each turn
    # ---- kept across turns of the same thread (checkpointer) ----
    history: Annotated[list[dict], operator.add]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_CITE = re.compile(r"\[(\d+(?:\s*[,、]\s*\d+)*)\]")


def renumber_citations(answer: str, n_passages: int) -> tuple[str, list[int], int]:
    """Drop citation numbers the LLM invented and renumber the rest 1..m in order of
    first appearance, so the user never sees "[1][4]" with [2] and [3] missing.

    Returns (answer, order, removed): `order[m-1]` is the original passage number of [m],
    and `removed` counts invented references, which the inspection reports as a defect."""
    order: list[int] = []
    removed = 0

    def _sub(m: re.Match) -> str:
        nonlocal removed
        new = []
        for raw in re.split(r"\s*[,、]\s*", m.group(1)):
            old = int(raw)
            if not 1 <= old <= n_passages:
                removed += 1  # hallucinated reference → removed and counted
                continue
            if old not in order:
                order.append(old)
            new.append(f"[{order.index(old) + 1}]")
        return "".join(dict.fromkeys(new))

    return _CITE.sub(_sub, answer).strip(), order, removed


def inspect_answer(
    cited: list[int], removed: int, numeric_issues: list[str], grounded: bool, n_unsupported: int
) -> list[dict]:
    """Answer inspection sheet (検品表). Same idea as visual inspection before shipping:
    every answer passes 4 checks, and any failed check downgrades the status."""
    return [
        {
            "name": "出典の付与",
            "passed": bool(cited),
            "detail": f"{len(cited)}件の出典を引用" if cited else "出典がありません",
        },
        {
            "name": "出典番号の妥当性",
            "passed": removed == 0,
            "detail": "存在しない出典番号なし" if not removed else f"存在しない出典番号 {removed}件を検出・除去",
        },
        {
            "name": "数値照合（ルール）",
            "passed": not numeric_issues,
            "detail": "全ての数値が出典と一致" if not numeric_issues else f"出典と不一致 {len(numeric_issues)}件",
        },
        {
            "name": "事実確認（LLM）",
            "passed": grounded,
            "detail": "全ての主張が出典で裏付け" if grounded else f"裏付けのない主張 {n_unsupported}件",
        },
    ]


def _clean_queries(queries: list[str], fallback: str) -> list[str]:
    cleaned = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))[:3]
    return cleaned or [fallback]


def _turn(state: "AgentState", answer: str) -> list[dict]:
    return [{"question": state["standalone_question"], "answer": answer}]


NodeFn = Callable[[AgentState, RunnableConfig], tuple[dict, str]]


def _timed(node: str, fn: NodeFn) -> Callable[[AgentState, RunnableConfig], dict]:
    """Adds a trace step (label, detail, elapsed ms) to every node's update."""

    def wrapper(state: AgentState, config: RunnableConfig) -> dict:
        t0 = time.perf_counter()
        update, detail = fn(state, config)
        ms = int((time.perf_counter() - t0) * 1000)
        step = {"node": node, "label": NODE_LABELS[node], "detail": detail, "ms": ms}
        update["trace"] = [*state.get("trace", []), step]
        return update

    return wrapper


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph(llm: Any, retriever: HybridRetriever, settings: Settings, checkpointer: Any = None):
    """`llm` is any LangChain chat model supporting `.with_structured_output()`;
    tests inject a fake one, so the whole agent runs offline in CI.
    Pass a checkpointer (e.g. `InMemorySaver()`) to enable multi-turn conversations."""

    def structured(schema):
        return llm.with_structured_output(schema, method="function_calling")

    planner = structured(QueryPlan)
    grader = structured(RelevanceGrade)
    writer = structured(DraftAnswer)
    checker = structured(GroundednessCheck)

    # ---- nodes (the config is passed on so callbacks such as token counting see every call) ---- #
    def analyze(state, config):
        history = (state.get("history") or [])[-settings.max_history_turns :] if settings.max_history_turns else []
        user_msg = state["question"]
        if history:
            user_msg = f"<history>\n{prompts.format_history(history)}\n</history>\n\n現在の質問: {user_msg}"
        plan: QueryPlan = planner.invoke([SystemMessage(prompts.ANALYZE_SYSTEM), HumanMessage(user_msg)], config)
        standalone = plan.standalone_question.strip() or state["question"]
        queries = _clean_queries(plan.search_queries, standalone)
        update = {
            "language": plan.language,
            "standalone_question": standalone,
            "search_queries": queries,
            "tried_queries": queries,
            "rewrites": 0,
        }
        detail = " / ".join(queries)
        if standalone != state["question"]:
            detail = f"「{standalone}」と解釈 → {detail}"
        return update, detail

    def retrieve(state, config):
        doc_ids = set(state["doc_ids"]) if state.get("doc_ids") else None
        hits = retriever.search(state["search_queries"], doc_ids=doc_ids)
        used_dense = any(h.dense_rank for h in hits)  # False if the embedding call failed → BM25 fallback
        mode = (
            "BM25 + Embedding (RRF)"
            if used_dense
            else "BM25" + ("（Embedding停止中のため代替）" if retriever.dense_enabled else "")
        )
        return {"candidates": hits}, f"{len(hits)}件の候補を取得 – {mode}"

    def grade(state, config):
        hits = state.get("candidates") or []
        if not hits:
            return {"context": []}, "候補なし"
        chunks = [h.chunk for h in hits]
        result: RelevanceGrade = grader.invoke(
            [
                SystemMessage(prompts.GRADE_SYSTEM),
                HumanMessage(
                    f"質問: {state['standalone_question']}\n\n"
                    f"<passages>\n{prompts.format_passages(chunks)}\n</passages>"
                ),
            ],
            config,
        )
        keep = {i for i in result.relevant_ids if 1 <= i <= len(chunks)}
        context = [c for i, c in enumerate(chunks, 1) if i in keep][: settings.context_k]
        return {"context": context}, f"{len(chunks)}件中 {len(context)}件を根拠として採用"

    def rewrite(state, config):
        tried = state.get("tried_queries", [])
        plan: QueryPlan = planner.invoke(
            [
                SystemMessage(prompts.REWRITE_SYSTEM),
                HumanMessage(f"質問: {state['standalone_question']}\n失敗したクエリ: {', '.join(tried)}"),
            ],
            config,
        )
        queries = _clean_queries(plan.search_queries, state["standalone_question"])
        return (
            {"search_queries": queries, "tried_queries": tried + queries, "rewrites": state.get("rewrites", 0) + 1},
            " / ".join(queries),
        )

    def generate(state, config):
        context = state["context"]
        lang = "英語で回答する" if state.get("language") == "en" else "日本語で回答する"
        draft: DraftAnswer = writer.invoke(
            [
                SystemMessage(prompts.GENERATE_SYSTEM.format(language_instruction=lang)),
                HumanMessage(
                    f"<passages>\n{prompts.format_passages(context)}\n</passages>\n\n"
                    f"質問: {state['standalone_question']}"
                ),
            ],
            config,
        )
        answer, order, removed = renumber_citations(draft.answer, len(context))
        detail = f"{len(order)}件の出典を引用"
        if removed:
            detail += f"（存在しない出典番号 {removed}件を除去）"
        return {"answer": answer, "cited_ids": order, "removed_citations": removed}, detail

    def verify(state, config):
        context, cited = state["context"], state.get("cited_ids") or []
        # Show the judge only the passages the answer cites, numbered like the answer's [1]..[m]
        # (cited[m-1] is the original position, see renumber_citations). A claim that is true but
        # attached to the wrong passage is therefore flagged too: this checks citation faithfulness.
        evidence = [context[i - 1] for i in cited] or context
        check: GroundednessCheck = checker.invoke(
            [
                SystemMessage(prompts.VERIFY_SYSTEM),
                HumanMessage(
                    f"<passages>\n{prompts.format_passages(evidence)}\n</passages>\n\n回答:\n{state['answer']}"
                ),
            ],
            config,
        )
        numeric = check_numbers(state["answer"], context, cited)
        unsupported = [] if check.grounded else check.unsupported_claims
        checks = inspect_answer(cited, state.get("removed_citations", 0), numeric, check.grounded, len(unsupported))
        passed = sum(c["passed"] for c in checks)
        failed = [c["name"] for c in checks if not c["passed"]]
        detail = f"{passed}/{len(checks)} 項目合格" + (f" – 要確認: {'・'.join(failed)}" if failed else "")
        return (
            {
                "status": "answered" if not failed else "partially_supported",
                "unsupported_claims": unsupported,
                "numeric_issues": numeric,
                "checks": checks,
                "history": _turn(state, state["answer"]),
            },
            detail,
        )

    def no_answer(state, config):
        lang = state.get("language", "ja")
        answer = prompts.NO_ANSWER.get(lang, prompts.NO_ANSWER["ja"])
        return (
            {
                "answer": answer,
                "status": "insufficient_evidence",
                "cited_ids": [],
                "context": [],
                "unsupported_claims": [],
                "numeric_issues": [],
                "removed_citations": 0,
                "checks": [],
                "history": _turn(state, answer),
            },
            f"{state.get('rewrites', 0) + 1}回検索しても根拠が見つからないため回答を控えました",
        )

    # ---- routing ---------------------------------------------------------- #
    def after_grade(state) -> str:
        if state.get("context"):
            return "generate"
        if state.get("rewrites", 0) < settings.max_query_rewrites:
            return "rewrite"
        return "no_answer"

    g = StateGraph(AgentState)
    for name, fn in [
        ("analyze", analyze),
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("verify", verify),
        ("no_answer", no_answer),
    ]:
        g.add_node(name, _timed(name, fn))

    g.add_edge(START, "analyze")
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "no_answer"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    g.add_edge("no_answer", END)
    return g.compile(checkpointer=checkpointer)
