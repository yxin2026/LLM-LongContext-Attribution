from lmaf.data.pac import (
    adapt_external_pac_sample,
    generate_pac_b_interference,
    generate_pac_c_overlap,
    generate_pac_d_multihop,
    is_external_pac_sample,
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


def test_external_pac_a_is_adapted_to_project_schema() -> None:
    row = {
        "sample_id": "A_com_4000_p10_0000",
        "subset": "A",
        "domain": "computer_science",
        "total_length": 4000,
        "total_length_unit": "tokens",
        "position_ratio": 0.1,
        "context": "RoPE was proposed by Su et al.",
        "question": "Who proposed RoPE?",
        "answer": "Su et al.",
    }

    sample = adapt_external_pac_sample(row)

    assert is_external_pac_sample(row)
    assert sample["experiment"] == "pac"
    assert sample["subtask"] == "A_position"
    assert sample["length_tokens_target"] == 4000
    assert sample["length_tokens_actual"] == 4000
    assert sample["position_percent"] == 10
    assert "Who proposed RoPE?" in sample["prompt"]
    assert "[Context]" in sample["prompt"]


def test_external_pac_b_density_and_interference_are_normalized() -> None:
    row = {
        "sample_id": "B_in__d25_8000_0000",
        "subset": "B",
        "total_length": 8000,
        "total_length_unit": "tokens",
        "dilution_type": "in_domain_related",
        "noise_density": 0.25,
        "context": "The answer is alpha.",
        "question": "What is the answer?",
        "answer": "alpha",
    }

    sample = adapt_external_pac_sample(row)

    assert sample["subtask"] == "B_interference"
    assert sample["density"] == 25
    assert sample["interference_type"] == "in_domain"
    assert sample["interference_type_raw"] == "in_domain_related"


def test_external_pac_d_keeps_hop_metadata() -> None:
    row = {
        "sample_id": "D_per_2h_near_8000_0000",
        "subset": "D",
        "total_length": 8000,
        "total_length_unit": "tokens",
        "chain_type": "person_relationship",
        "num_hops": 2,
        "distance_level": "near",
        "fact_chain": [
            {"entity": "Alice", "relation": "mentor", "target": "Bob"},
            {"entity": "Bob", "relation": "collaborator", "target": "Carol"},
        ],
        "context": "Alice's mentor is Bob. Bob's collaborator is Carol.",
        "question": "Who is Alice's mentor's collaborator?",
        "answer": "Carol",
    }

    sample = adapt_external_pac_sample(row)

    assert sample["subtask"] == "D_multihop"
    assert sample["hops"] == 2
    assert sample["hop_distance"] == "near"
    assert sample["chain_type"] == "person_relationship"
    assert sample["intermediate_answers"] == ["Bob"]
