from app.ingest import Chunk
from eval.fault_injection import build_samples, confusion_metrics, run


def test_one_digit_errors_are_created_and_caught_by_the_rule_check():
    chunk = Chunk("d:p1:c0", "d", "白書", "省", 1, "学校図書館図書の整備に単年度199億円を措置している。")
    samples = build_samples([chunk], [{"doc_id": "d", "pages": [1]}])

    assert len(samples) == 1
    assert samples[0].correct != samples[0].corrupted  # exactly one digit was changed
    result = run(samples)["Rule-based number check"]
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 0, 0, 1)
    assert result["precision"] == result["recall"] == result["f1"] == 1.0
    assert result["false_alarm"] == 0.0


def test_confusion_metrics_with_a_noisy_verifier():
    # 4 samples: flags 3 of 4 corrupted sentences and 1 of 4 correct sentences
    m = confusion_metrics([(False, True), (False, True), (True, True), (False, False)])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 3)
    assert m["precision"] == m["recall"] == m["f1"] == 0.75
    assert m["false_alarm"] == 0.25


def test_hard_samples_swap_in_a_number_from_the_same_passage_and_slip_past_the_rule_check():
    chunk = Chunk("d:p1:c0", "d", "白書", "省", 1, "人口は2008年をピークに減少に転じ、2022年には1億2,500万人となった。")
    samples = build_samples([chunk], [{"doc_id": "d", "pages": [1]}], hard=True)

    assert samples and all(s.correct != s.corrupted for s in samples)
    result = run([], hard_samples=samples)["Rule-based number check (hard)"]
    assert result["recall"] == 0.0  # the documented blind spot: every number exists somewhere in the passage
    assert result["false_alarm"] == 0.0
