from lmaf.data.niah import generate_multi_niah, generate_single_niah, validate_token_length
from lmaf.utils.token_count import TokenCounter


def test_single_niah_contains_answer_and_tracks_position() -> None:
    counter = TokenCounter()
    sample = generate_single_niah(512, 50, seed=42, sample_index=0, counter=counter)
    assert sample["answer"] in sample["prompt"]
    assert sample["prompt"].count(sample["answer"]) == 1
    validation = validate_token_length(sample, counter)
    assert validation["length_error_ratio"] < 0.12
    assert validation["position_error_abs"] <= 3.0


def test_multi_niah_generates_three_needles() -> None:
    counter = TokenCounter()
    sample = generate_multi_niah(768, "uniform", seed=42, sample_index=1, counter=counter)
    assert len(sample["needles"]) == 3
    for needle in sample["needles"]:
        assert needle in sample["prompt"]

