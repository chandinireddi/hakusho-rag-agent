from app.grounding import check_numbers, extract_numbers
from app.ingest import Chunk


def _ctx(*texts):
    return [Chunk(f"d:p1:c{i}", "d", "白書", "省", 1, t) for i, t in enumerate(texts)]


def test_number_formats_are_normalised():
    assert extract_numbers("１万７,１２９教室、20.0%、2023年") == {"1", "7129", "20", "2023"}


def test_matching_numbers_pass():
    ctx = _ctx("学校図書館図書の整備に単年度199億円を措置。")
    assert check_numbers("単年度199億円です[1]。", ctx, cited_ids=[1]) == []


def test_one_wrong_digit_is_caught():
    ctx = _ctx("学校図書館図書の整備に単年度199億円を措置。")
    issues = check_numbers("単年度190億円です[1]。", ctx, cited_ids=[1])
    assert len(issues) == 1 and "190" in issues[0]


def test_number_without_citation_is_flagged():
    issues = check_numbers("人口は2008年にピークを迎えました。", _ctx("2008年"), cited_ids=[])
    assert issues and issues[0].startswith("出典のない数値")


def test_citation_points_to_the_renumbered_passage():
    ctx = _ctx("COP15はモントリオール。", "人口は2008年がピーク。")
    # the answer cites [1], which is passage #2 after renumbering
    assert check_numbers("2008年がピークです[1]。", ctx, cited_ids=[2]) == []


def test_list_markers_are_not_treated_as_numbers():
    assert check_numbers("1. モントリオールで開催[1]。", _ctx("モントリオール"), cited_ids=[1]) == []
