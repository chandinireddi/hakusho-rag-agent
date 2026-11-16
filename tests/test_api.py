import json
import tomllib

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.config import ROOT
from app.graph import build_graph
from app.main import Components, create_app
from app.schemas import DraftAnswer, GroundednessCheck, QueryPlan, RelevanceGrade


@pytest.fixture
def client(fake_llm, retriever, settings):
    fake_llm.queue(QueryPlan, QueryPlan(language="ja", search_queries=["Rapidus 拠点"]))
    fake_llm.queue(RelevanceGrade, RelevanceGrade(relevant_ids=[1]))
    fake_llm.queue(DraftAnswer, DraftAnswer(answer="北海道です[1]。"))
    fake_llm.queue(GroundednessCheck, GroundednessCheck(grounded=True, unsupported_claims=[]))
    app = create_app(Components(settings, retriever, build_graph(fake_llm, retriever, settings)))
    with TestClient(app) as c:
        yield c


def test_ask_returns_answer_with_page_level_citation(client):
    res = client.post("/api/ask", json={"question": "Rapidusの製造拠点はどこ？"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "answered"
    cite = body["citations"][0]
    assert (cite["ref"], cite["doc_id"], cite["page"]) == (1, "meti", 17)
    assert cite["url"] == "/documents/meti.pdf#page=17"
    assert body["thread_id"]  # returned so the client can ask a follow-up in the same thread
    assert body["numeric_issues"] == []
    assert [c["passed"] for c in body["checks"]] == [True] * 4  # inspection sheet is part of the API


def test_stream_emits_steps_then_result(client):
    with client.stream("POST", "/api/ask/stream", json={"question": "Rapidusの製造拠点はどこ？"}) as res:
        events = [line.removeprefix("event: ") for line in res.iter_lines() if line.startswith("event: ")]
    assert events[:3] == ["step", "step", "step"]
    assert events[-1] == "result"


@pytest.mark.parametrize("question", ["", "あ", "x" * 501])
def test_invalid_questions_are_rejected(client, question):
    assert client.post("/api/ask", json={"question": question}).status_code == 422


def test_unknown_document_filter_is_rejected(client):
    res = client.post("/api/ask", json={"question": "Rapidusの製造拠点はどこ？", "doc_ids": ["no-such-doc"]})
    assert res.status_code == 422


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["chunks"] == 4 and body["dense_retrieval"] is False


def test_stream_result_is_valid_json(client):
    with client.stream("POST", "/api/ask/stream", json={"question": "Rapidusの製造拠点はどこ？"}) as res:
        data = [line[6:] for line in res.iter_lines() if line.startswith("data: ")]
    assert json.loads(data[-1])["answer"] == "北海道です[1]。"


def test_version_is_the_same_everywhere(client):
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__ == client.get("/api/health").json()["version"]
