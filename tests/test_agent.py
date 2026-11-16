from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph, renumber_citations
from app.schemas import DraftAnswer, GroundednessCheck, QueryPlan, RelevanceGrade


def _ask(graph, question, thread_id=None):
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    return graph.invoke({"question": question, "trace": []}, config)


def test_renumber_removes_hallucinated_refs_and_closes_gaps():
    text, order, removed = renumber_citations("A[3]。B[1, 9]。C[3]。", n_passages=3)
    assert text == "A[1]。B[2]。C[1]。"
    assert order == [3, 1]
    assert removed == 1  # [9] does not exist → counted as a defect


def test_happy_path_answer_with_citation(fake_llm, retriever, settings):
    fake_llm.queue(QueryPlan, QueryPlan(language="ja", search_queries=["COP15 第二部 開催地"]))
    fake_llm.queue(RelevanceGrade, RelevanceGrade(relevant_ids=[1]))
    fake_llm.queue(DraftAnswer, DraftAnswer(answer="2022年12月にカナダ・モントリオールで開催されました[1]。"))
    fake_llm.queue(GroundednessCheck, GroundednessCheck(grounded=True, unsupported_claims=[]))

    state = _ask(build_graph(fake_llm, retriever, settings), "COP15第二部の開催地は？")

    assert state["status"] == "answered"
    assert state["numeric_issues"] == []
    assert state["context"][0].doc_id == "env"
    assert len(state["checks"]) == 4 and all(c["passed"] for c in state["checks"])
    assert [t["node"] for t in state["trace"]] == ["analyze", "retrieve", "grade", "generate", "verify"]


def test_agent_rewrites_query_then_refuses_instead_of_hallucinating(fake_llm, retriever, settings):
    fake_llm.queue(
        QueryPlan,
        QueryPlan(language="ja", search_queries=["大阪万博 来場者数"]),
        QueryPlan(language="ja", search_queries=["万博 入場者"]),
    )
    fake_llm.queue(RelevanceGrade, RelevanceGrade(relevant_ids=[]), RelevanceGrade(relevant_ids=[]))

    state = _ask(build_graph(fake_llm, retriever, settings), "大阪万博の来場者数は？")

    assert state["status"] == "insufficient_evidence"
    assert "DraftAnswer" not in fake_llm.calls  # never generated from weak context
    assert [t["node"] for t in state["trace"]].count("retrieve") == 2


def test_unsupported_claims_and_wrong_numbers_are_flagged(fake_llm, retriever, settings):
    fake_llm.queue(QueryPlan, QueryPlan(language="en", search_queries=["人口 減少"]))
    fake_llm.queue(RelevanceGrade, RelevanceGrade(relevant_ids=[1]))
    fake_llm.queue(DraftAnswer, DraftAnswer(answer="Peaked in 2008 [1] and will halve by 2050."))
    fake_llm.queue(GroundednessCheck, GroundednessCheck(grounded=False, unsupported_claims=["halve by 2050"]))

    state = _ask(build_graph(fake_llm, retriever, settings), "When did population peak?")

    assert state["status"] == "partially_supported"
    assert state["unsupported_claims"] == ["halve by 2050"]
    assert "2050" in state["numeric_issues"][0]  # caught by the deterministic check as well


def test_wrong_number_downgrades_status_even_if_llm_judge_says_ok(fake_llm, retriever, settings):
    fake_llm.queue(QueryPlan, QueryPlan(language="ja", search_queries=["人口 ピーク"]))
    fake_llm.queue(RelevanceGrade, RelevanceGrade(relevant_ids=[1]))
    fake_llm.queue(DraftAnswer, DraftAnswer(answer="人口は2009年をピークに減少しました[1]。"))
    fake_llm.queue(GroundednessCheck, GroundednessCheck(grounded=True, unsupported_claims=[]))

    state = _ask(build_graph(fake_llm, retriever, settings), "人口のピークは？")

    assert state["status"] == "partially_supported"
    assert "2009" in state["numeric_issues"][0]


def test_invented_citation_number_fails_the_inspection(fake_llm, retriever, settings):
    fake_llm.queue(QueryPlan, QueryPlan(language="ja", search_queries=["COP15 開催地"]))
    fake_llm.queue(RelevanceGrade, RelevanceGrade(relevant_ids=[1]))
    fake_llm.queue(DraftAnswer, DraftAnswer(answer="モントリオールで開催されました[1][7]。"))
    fake_llm.queue(GroundednessCheck, GroundednessCheck(grounded=True, unsupported_claims=[]))

    state = _ask(build_graph(fake_llm, retriever, settings), "COP15第二部の開催地は？")

    assert state["answer"] == "モントリオールで開催されました[1]。"  # [7] removed from the answer
    assert state["status"] == "partially_supported"
    failed = [c["name"] for c in state["checks"] if not c["passed"]]
    assert failed == ["出典番号の妥当性"]


def test_follow_up_question_uses_conversation_memory(fake_llm, retriever, settings):
    graph = build_graph(fake_llm, retriever, settings, checkpointer=InMemorySaver())
    rewritten = "Rapidusはどの半導体を製造しますか？"
    fake_llm.queue(
        QueryPlan,
        QueryPlan(language="ja", search_queries=["Rapidus 拠点"]),
        QueryPlan(language="ja", standalone_question=rewritten, search_queries=["Rapidus 半導体"]),
    )
    fake_llm.queue(RelevanceGrade, RelevanceGrade(relevant_ids=[1]), RelevanceGrade(relevant_ids=[1]))
    fake_llm.queue(DraftAnswer, DraftAnswer(answer="北海道です[1]。"), DraftAnswer(answer="次世代半導体です[1]。"))
    fake_llm.queue(GroundednessCheck, *[GroundednessCheck(grounded=True, unsupported_claims=[])] * 2)

    _ask(graph, "Rapidusの拠点はどこ？", thread_id="t1")
    state = _ask(graph, "そこでは何を作る？", thread_id="t1")

    second_plan_prompt = [m for name, m in fake_llm.messages if name == "QueryPlan"][1][1].content
    assert "Rapidusの拠点はどこ？" in second_plan_prompt  # history was given to the planner
    assert state["standalone_question"] == rewritten
    assert len(state["history"]) == 2
    assert [t["node"] for t in state["trace"]][0] == "analyze"  # trace was reset for the new turn
    assert len(state["trace"]) == 5


def test_verifier_sees_only_the_cited_passages_numbered_like_the_answer(fake_llm, retriever, settings):
    fake_llm.queue(QueryPlan, QueryPlan(language="ja", search_queries=["モントリオール 人口 ピーク"]))
    fake_llm.queue(RelevanceGrade, RelevanceGrade(relevant_ids=[1, 2]))
    fake_llm.queue(DraftAnswer, DraftAnswer(answer="日本の人口は2008年をピークに減少に転じました[2]。"))
    fake_llm.queue(GroundednessCheck, GroundednessCheck(grounded=True, unsupported_claims=[]))

    state = _ask(build_graph(fake_llm, retriever, settings), "人口のピークは？")

    judged = next(m for name, m in fake_llm.messages if name == "GroundednessCheck")[1].content
    cited_chunk = state["context"][state["cited_ids"][0] - 1]
    assert cited_chunk.text in judged
    assert '<passage id="2"' not in judged  # the uncited passage is not shown to the judge
    assert '<passage id="1"' in judged  # the cited one is renumbered to match the answer's [1]
    assert len(state["context"]) == 2
