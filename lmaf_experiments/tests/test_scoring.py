from lmaf.eval.metrics import exact_match, normalize_answer, token_f1
from lmaf.eval.scorers import classify_error, score_niah


def test_normalize_removes_punctuation_and_space() -> None:
    assert normalize_answer(" Code-123, ") == "code123"


def test_qa_scores_multiple_answers() -> None:
    assert exact_match("Aurora", ["Borealis", "Aurora"]) == 1.0
    assert token_f1("the final answer", "final answer") > 0.5


def test_niah_contains_answer_with_explanation() -> None:
    scored = score_niah("The code is ABC-123.", "ABC-123")
    assert scored["score"] == 1.0
    assert scored["contains_answer"] == 1


def test_classify_error_labels_empty_and_distractor() -> None:
    assert classify_error("", "abc", ["xyz"]) == "refusal_or_empty"
    assert classify_error("xyz", "abc", ["xyz"]) == "confused_with_distractor"

