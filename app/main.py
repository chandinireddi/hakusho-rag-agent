import json
import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.callbacks import UsageMetadataCallbackHandler

from app import __version__
from app.catalog import get_meta
from app.config import ROOT, Settings, get_settings
from app.graph import build_graph
from app.ingest import load_corpus
from app.retriever import HybridRetriever
from app.schemas import AskRequest, AskResponse, Citation, QualityCheck, TraceStep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
logger = logging.getLogger("app")

STATIC_DIR = ROOT / "static"
AI_ERROR = "AIサービスとの通信に失敗しました。時間をおいて再度お試しください。"


@dataclass
class Components:
    settings: Settings
    retriever: HybridRetriever
    graph: Any
    llm: Any = None  # exposed for the evaluation (verifier fault-injection benchmark)


def enable_tracing(settings: Settings) -> bool:
    """Opt-in LangSmith tracing. pydantic-settings reads `.env` into `Settings` only, so the
    variables LangChain looks for are exported here. Values already in the environment win."""
    if not settings.langsmith_api_key:
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    return True


def build_components(settings: Settings, *, memory: bool = True) -> Components:
    """Production wiring: OpenAI chat model + OpenAI embeddings (+ in-memory conversation store)."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    if enable_tracing(settings):
        logger.info("LangSmith tracing enabled (project: %s)", settings.langsmith_project)

    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, timeout=30, max_retries=2)

    def embed(texts: list[str]) -> list[list[float]]:
        res = client.embeddings.create(model=settings.embedding_model, input=texts)
        return [d.embedding for d in res.data]

    chunks, fingerprint = load_corpus(settings)
    retriever = HybridRetriever(chunks, settings, embed_fn=embed, fingerprint=fingerprint)
    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
        timeout=30,
        max_retries=2,
    )
    checkpointer = InMemorySaver() if memory else None
    return Components(settings, retriever, build_graph(llm, retriever, settings, checkpointer=checkpointer), llm)


# --------------------------------------------------------------------------- #
# Request / result helpers
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    """Everything one request needs: input state, LangGraph config and a token counter."""

    thread_id: str
    state: dict
    config: dict
    usage: UsageMetadataCallbackHandler
    started: float

    @classmethod
    def start(cls, req: AskRequest) -> "Run":
        thread_id = req.thread_id or uuid.uuid4().hex
        usage = UsageMetadataCallbackHandler()
        state = {"question": req.question.strip(), "doc_ids": req.doc_ids, "trace": []}
        config = {"callbacks": [usage], "configurable": {"thread_id": thread_id}}
        return cls(thread_id, state, config, usage, time.perf_counter())

    def total_tokens(self) -> int:
        return sum(u.get("total_tokens", 0) for u in self.usage.usage_metadata.values())


def to_response(req: AskRequest, run: Run, state: dict, settings: Settings) -> AskResponse:
    context = state.get("context") or []
    citations = []
    for new_ref, old_ref in enumerate(state.get("cited_ids") or [], start=1):
        c = context[old_ref - 1]
        citations.append(
            Citation(
                ref=new_ref,
                doc_id=c.doc_id,
                title=c.title,
                ministry=c.ministry,
                page=c.page,
                snippet=c.text[:220] + ("…" if len(c.text) > 220 else ""),
                url=f"/documents/{c.doc_id}.pdf#page={c.page}",
            )
        )
    return AskResponse(
        question=req.question,
        standalone_question=state.get("standalone_question", req.question),
        answer=state.get("answer", ""),
        status=state.get("status", "insufficient_evidence"),
        citations=citations,
        checks=[QualityCheck(**c) for c in state.get("checks") or []],
        unsupported_claims=state.get("unsupported_claims") or [],
        numeric_issues=state.get("numeric_issues") or [],
        search_queries=state.get("tried_queries") or [],
        trace=[TraceStep(**t) for t in state.get("trace", [])],
        thread_id=run.thread_id,
        model=settings.chat_model,
        total_tokens=run.total_tokens(),
        latency_ms=int((time.perf_counter() - run.started) * 1000),
    )


def _check_doc_ids(req: AskRequest, retriever: HybridRetriever) -> None:
    unknown = set(req.doc_ids or []) - {c.doc_id for c in retriever.chunks}
    if unknown:
        raise HTTPException(422, f"Unknown doc_ids: {sorted(unknown)}")


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(components: Components | None = None) -> FastAPI:
    settings = components.settings if components else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.c = components or build_components(settings)
        r = app.state.c.retriever
        logger.info("Ready: %d chunks, dense=%s, model=%s", len(r.chunks), r.dense_enabled, settings.chat_model)
        yield

    app = FastAPI(title="White Paper RAG Agent", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/documents", StaticFiles(directory=settings.documents_dir), name="documents")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health(request: Request):
        r = request.app.state.c.retriever
        return {
            "status": "ok",
            "chunks": len(r.chunks),
            "documents": len({c.doc_id for c in r.chunks}),
            "dense_retrieval": r.dense_enabled,
            "tracing": os.environ.get("LANGSMITH_TRACING") == "true",
            "model": settings.chat_model,
            "version": __version__,
        }

    @app.get("/api/documents")
    def documents(request: Request):
        r = request.app.state.c.retriever
        pages: dict[str, int] = {}
        for c in r.chunks:
            pages[c.doc_id] = max(pages.get(c.doc_id, 0), c.page)
        return [
            {"doc_id": d, "title": get_meta(d).title, "ministry": get_meta(d).ministry, "pages": n}
            for d, n in pages.items()
        ]

    @app.post("/api/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request):
        c = request.app.state.c
        _check_doc_ids(req, c.retriever)
        run = Run.start(req)
        try:
            state = c.graph.invoke(run.state, run.config)
        except Exception as exc:  # OpenAI outage, timeout, …
            logger.exception("Agent failed")
            raise HTTPException(502, AI_ERROR) from exc
        return to_response(req, run, state, settings)

    @app.post("/api/ask/stream")
    def ask_stream(req: AskRequest, request: Request):
        """Server-Sent Events: one `step` event per graph node, then a `result` event."""
        c = request.app.state.c
        _check_doc_ids(req, c.retriever)
        run = Run.start(req)

        def events() -> Iterator[str]:
            state, sent = dict(run.state), 0
            try:
                for update in c.graph.stream(run.state, run.config, stream_mode="updates"):
                    for delta in update.values():
                        state.update(delta)
                        for step in state["trace"][sent:]:  # each node returns the full trace list
                            yield _sse("step", step)
                        sent = len(state["trace"])
                yield _sse("result", to_response(req, run, state, settings).model_dump())
            except Exception:
                logger.exception("Agent failed (stream)")
                yield _sse("error", {"detail": AI_ERROR})

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    return app


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


app = create_app()
