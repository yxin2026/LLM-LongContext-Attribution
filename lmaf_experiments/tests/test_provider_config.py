from lmaf.inference.client import resolve_provider_model


def test_siliconflow_alias_resolution() -> None:
    assert resolve_provider_model("siliconflow", "qwen35_9b") == "Qwen/Qwen3.5-9B"
    assert resolve_provider_model("siliconflow", "qwen3_8b") == "Qwen/Qwen3-8B"
    assert resolve_provider_model("siliconflow", "hunyuan_a13b") == "tencent/Hunyuan-A13B-Instruct"
    assert resolve_provider_model("siliconflow", "Qwen/Qwen3.5-9B") == "Qwen/Qwen3.5-9B"


def test_removed_models_are_not_aliases() -> None:
    assert resolve_provider_model("siliconflow", "deepseek_r1_distill_qwen_14b") == "deepseek_r1_distill_qwen_14b"
    assert resolve_provider_model("siliconflow", "gemma4_26b_a4b") == "gemma4_26b_a4b"
    assert resolve_provider_model("siliconflow", "gemma4_31b") == "gemma4_31b"


def test_local_model_name_is_not_changed() -> None:
    assert resolve_provider_model("local", "qwen35_9b") == "qwen35_9b"
