import pytest

from utils.model_metadata import (
    anthropic_effort_config,
    anthropic_model_flags,
    anthropic_reasoning_effort_config,
    clamp_openai_reasoning_effort,
    flux_sdcpp_valid_text_encoder_quant,
    flux_valid_backend,
    get_max_tokens_cap,
    is_anthropic_no_sampling_model,
    is_gemini_3_plus_model,
    is_gemini_4_model,
    is_gemini_no_sampling_model,
    is_google_reasoning_model,
    is_opencode_chat_completions_model,
    is_opencode_multimodal_model,
    is_rosetta_model,
    is_sonnet_5_model,
    is_sonnet_55_model,
    is_xai_reasoning_model,
    resolve_openai_api_model_name,
    supports_gpt5_max_effort,
    supports_gpt5_xhigh_effort,
    supports_meta_max_effort,
    supports_openai_original_image_detail,
    supports_xai_reasoning_parameter,
    xai_reasoning_efforts,
)


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("OpenAI", "gpt-5.2-chat-latest", 16384),
        ("OpenAI", "gpt-5.5", None),
        ("OpenRouter", "openai/gpt-5.2-chat", 16384),
        ("OpenRouter", "anthropic/claude-sonnet-4-6", None),
        ("Moonshot AI", "kimi-k2.5", 32768),
        ("Google", "gemini-4-argon", None),
        ("Google", "gemini-3.8-flash", None),
        ("OpenRouter", "google/gemini-4-argon", None),
        ("OpenAI", None, None),
    ],
)
def test_get_max_tokens_cap(provider, model, expected):
    assert get_max_tokens_cap(provider, model) == expected


@pytest.mark.parametrize(
    ("model", "xhigh", "max_"),
    [
        ("gpt-5", False, False),
        ("gpt-5.2", True, False),
        ("gpt-5.6-sol", True, True),
        # Minor versions must compare numerically, not lexically.
        ("gpt-5.10", True, True),
        ("openai/gpt-5.4", True, False),
        ("claude-opus-5", False, False),
    ],
)
def test_gpt5_effort_support(model, xhigh, max_):
    assert supports_gpt5_xhigh_effort(model) is xhigh
    assert supports_gpt5_max_effort(model) is max_


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gpt-5", "max", "high"),
        ("gpt-5.2", "max", "xhigh"),
        ("gpt-5", "xhigh", "high"),
        ("gpt-5.6-sol", "max", "max"),
        ("gpt-5", "none", "none"),
        ("gpt-6-sol", "minimal", "low"),
        ("gpt-6-sol", "none", "none"),
        ("gpt-6-astra", "none", "low"),
        ("gpt-6.1-sol", "none", "low"),
        ("gpt-6-sol", "max", "max"),
    ],
)
def test_clamp_openai_reasoning_effort(model, effort, expected):
    assert clamp_openai_reasoning_effort(model, effort) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.6-sol-pro", "gpt-5.6-sol"),
        ("gpt-6-astra-pro", "gpt-6-astra"),
        ("gpt-5.5-pro", "gpt-5.5-pro"),
        ("gpt-5.4", "gpt-5.4"),
        (None, None),
    ],
)
def test_resolve_openai_api_model_name(model, expected):
    assert resolve_openai_api_model_name(model) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-6-luna", True),
        ("gpt-5.4", True),
        ("gpt-5.3", False),
        ("gpt-5.5-mini", False),
    ],
)
def test_supports_openai_original_image_detail(model, expected):
    assert supports_openai_original_image_detail(model) is expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-4-5-20251101", {"is_claude_effort": True}),
        (
            "claude-sonnet-4-6",
            {"is_claude_effort": True, "is_claude_effort_max": True},
        ),
        (
            "claude-opus-4-7",
            {
                "is_claude_effort": True,
                "is_claude_effort_max": True,
                "is_claude_effort_xhigh": True,
            },
        ),
        (
            "anthropic/claude-haiku-5-5",
            {
                "is_claude_effort": True,
                "is_claude_effort_max": True,
                "is_claude_effort_xhigh": True,
                "is_claude_adaptive_default": True,
            },
        ),
        (None, {}),
    ],
)
def test_anthropic_model_flags(model, expected):
    assert anthropic_model_flags(model) == expected


def test_sonnet_5_and_55_are_distinct():
    assert is_sonnet_55_model("claude-sonnet-5-5")
    assert not is_sonnet_5_model("claude-sonnet-5-5")
    assert is_sonnet_5_model("claude-sonnet-5")


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-4-7", True),
        ("claude-fable-5-1", True),
        ("claude-sonnet-4-6", False),
        ("claude-opus-4-5-20251101", False),
    ],
)
def test_is_anthropic_no_sampling_model(model, expected):
    assert is_anthropic_no_sampling_model(model) is expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-fable-5-1", (False, [], None)),
        ("claude-haiku-5-5", (True, ["auto", "none"], "auto")),
        ("claude-sonnet-4-6", (True, ["auto", "none"], "auto")),
        (
            "claude-opus-4-5-20251101",
            (True, ["high", "medium", "low", "none"], "low"),
        ),
        ("gpt-5.5", (False, [], None)),
    ],
)
def test_anthropic_reasoning_effort_config(model, expected):
    assert anthropic_reasoning_effort_config(model) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-4-5", (True, ["high", "medium", "low"], "high")),
        ("claude-sonnet-4-6", (True, ["max", "high", "medium", "low"], "high")),
        (
            "claude-opus-4-7",
            (True, ["max", "xhigh", "high", "medium", "low"], "high"),
        ),
    ],
)
def test_anthropic_effort_config(model, expected):
    assert anthropic_effort_config(model) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("grok-4.3", True),
        ("grok-4.20-reasoning", True),
        ("grok-4.20-non-reasoning", False),
        ("grok-3", False),
    ],
)
def test_is_xai_reasoning_model(model, expected):
    assert is_xai_reasoning_model(model) is expected


@pytest.mark.parametrize(
    ("model", "efforts", "takes_parameter"),
    [
        ("grok-4.7", ("low", "medium", "high", "xhigh"), True),
        ("grok-4-multi-agent", ("low", "medium", "high", "xhigh"), True),
        ("grok-4.5-fast", ("none", "low", "medium", "high"), True),
        # The efforts go by generation alone; only the parameter check sees the suffix
        ("grok-4.5-non-reasoning", ("none", "low", "medium", "high"), False),
        ("grok-4.20-reasoning", (), False),
        (None, (), False),
    ],
)
def test_xai_reasoning_efforts(model, efforts, takes_parameter):
    assert xai_reasoning_efforts(model) == efforts
    assert supports_xai_reasoning_parameter(model) is takes_parameter


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("muse-wonder-1.0", True),
        ("muse-spark-1.3", True),
        ("muse-spark-1.3-contributor", False),
        ("muse-spark-1.3-contributor-free", False),
        ("muse-spark-1.2", False),
        (None, False),
    ],
)
def test_supports_meta_max_effort(model, expected):
    assert supports_meta_max_effort(model) is expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("glm-5.2", False),
        ("opencode/deepseek-v4-pro", False),
        ("qwen3-coder", False),
        ("gpt-5.5", True),
        ("claude-sonnet-4-6", True),
        ("kimi-k2.5", True),
        ("some-brand-new-model", True),
    ],
)
def test_is_opencode_multimodal_model(model, expected):
    assert is_opencode_multimodal_model(model) is expected


@pytest.mark.parametrize(
    ("model", "tier", "expected"),
    [
        ("kimi-k2.6", "zen", True),
        ("glm-5.3-flash", "go", True),
        ("some-brand-new-model", "zen", True),
        ("gpt-6-luna", "go", False),
        ("claude-haiku-5-5", "zen", False),
        ("gemini-3.8-flash", "zen", False),
        ("grok-4.7", "go", False),
        ("muse-spark-1.3-contributor-free", "zen", False),
        ("jev-1.13-free", "zen", False),
        ("qwen3.7-plus", "zen", False),
        ("qwen3.8-max", "zen", True),
        ("qwen3.8-max", "go", False),
        ("minimax-m3", "zen", True),
        ("minimax-m3", "go", False),
    ],
)
def test_is_opencode_chat_completions_model(model, tier, expected):
    assert is_opencode_chat_completions_model(model, tier) is expected


@pytest.mark.parametrize("model", ["gemini-4-argon", "google/gemini-4-argon"])
def test_gemini_4_is_a_thinking_level_model_without_sampling(model):
    assert is_gemini_4_model(model)
    assert is_gemini_3_plus_model(model)
    assert is_gemini_no_sampling_model(model)
    assert is_google_reasoning_model(model)


def test_gemini_4_does_not_match_gemma_4_or_gemini_3():
    assert not is_gemini_4_model("gemma-4-31b-it")
    assert not is_gemini_4_model("gemini-3.8-flash")
    assert not is_gemini_3_plus_model("gemma-4-31b-it")


def test_is_rosetta_model_requires_vendor_and_name():
    assert is_rosetta_model("yanoljanext-rosetta-4b-2511")
    assert is_rosetta_model("YanoljaNEXT_Rosetta 12B")
    assert not is_rosetta_model("rosetta-4b")


def test_flux_valid_backend():
    assert flux_valid_backend("flux_klein_4b", "nunchaku") == "sdnq"
    assert flux_valid_backend("flux_klein_9b", "sdcpp") == "sdcpp"
    assert flux_valid_backend("flux_kontext", "nunchaku") == "nunchaku"
    assert flux_valid_backend("flux_kontext", "bogus") == "sdnq"


def test_flux_text_encoder_quant_falls_back_to_model_default():
    assert flux_sdcpp_valid_text_encoder_quant("flux_klein_4b", "Q8_0") == "Q8_0"
    # Qwen-only quant is invalid for Kontext's T5 encoder.
    assert flux_sdcpp_valid_text_encoder_quant("flux_kontext", "Q4_K_XL") == "Q4_K_M"
    assert flux_sdcpp_valid_text_encoder_quant("flux_klein_9b", "nope") == "Q4_K_XL"
