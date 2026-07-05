from lmaf.data.pac import (
    generate_pac_b_interference,
    generate_pac_c_overlap,
    generate_pac_d_multihop,
    score_pac_sample,
)
from lmaf.utils.token_count import TokenCounter


def test_pac_b_records_interference_metadata() -> None:
    sample = generate_pac_b_interference(768, 10, 50, "in_domain", seed=42, sample_index=0, counter=TokenCounter())
    assert sample["density"] == 50
    assert sample["interference_type"] == "in_domain"
    assert sample["answer"] in sample["prompt"]
    assert sample["distractor_answers"]


def test_pac_c_classifies_distractor_confusion() -> None:
    sample = generate_pac_c_overlap(768, "high", "near", seed=42, sample_index=0, counter=TokenCounter())
    distractor = sample["distractor_answers"][0]
    scored = score_pac_sample(sample, distractor)
    assert scored["error_type"] == "confused_with_distractor"
    assert scored["score"] == 0.0


def test_pac_d_multihop_has_intermediate_probes() -> None:
    sample = generate_pac_d_multihop(1024, 3, 1024, seed=42, sample_index=0, counter=TokenCounter())
    assert sample["answer"] in sample["prompt"]
    assert len(sample["intermediate_questions"]) == 3
    assert len(sample["intermediate_answers"]) == 3

